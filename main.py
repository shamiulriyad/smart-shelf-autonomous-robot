"""Warehouse pick-and-place prototype (computer vision + simulated robot).

  python main.py --pick 102                 # webcam demo: "Pick Product 102" (or --pick "Product B")
  python main.py                            # interactive: type any product (ID or name), the robot finds its shelf
  python main.py --synthetic                # interactive, on rendered markers (no webcam)
  python main.py --pick 102 --synthetic     # same flow on rendered markers, no webcam needed
  python main.py --calibrate --square-cm 2.5   # capture chessboard views and save camera calibration
  python main.py --show-map                 # print the room map and the planned path per product

Robot movement is simulated. The webcam only supplies marker detections; it does not localise the robot.
"""
import argparse
import logging
import sys

from aruco.calibration import (DEFAULT_CALIBRATION_PATH, CalibrationError, load_if_available,
                               run_interactive_capture)
from aruco.perception import Perception, UserAbort
from aruco.pose_estimator import PoseEstimator
from camera.camera_source import CameraError, SimulatedWorldCamera, WebcamSource, build_world_markers
from config.loader import DEFAULT_CONFIG_PATH, ConfigError, WarehouseConfig, load_config
from navigation.path_planner import PathPlanner
from robot.robot_controller import SimulatedRobot
from robot.state_machine import (MachineSettings, PickAndPlaceStateMachine, PlacementPreconditionError,
                                 State)

log = logging.getLogger("warehouse")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--pick", metavar="PRODUCT", help="product marker ID (e.g. 102) or name (e.g. 'Product B'); "
                      "omit all modes to be asked for a product interactively")
    mode.add_argument("--calibrate", action="store_true", help="interactive chessboard camera calibration")
    mode.add_argument("--show-map", action="store_true", help="print the room map and planned paths, then exit")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--calibration-file", default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, help="request this capture width (must match the calibration)")
    p.add_argument("--height", type=int, help="request this capture height")
    p.add_argument("--synthetic", action="store_true", help="use rendered marker frames instead of a webcam")
    p.add_argument("--no-display", action="store_true", help="no preview window")
    p.add_argument("--board", default="7x6", help="chessboard INNER corners, COLSxROWS (default 7x6)")
    p.add_argument("--square-cm", type=float, help="chessboard square size in cm (required for --calibrate)")
    return p.parse_args()


def show_map(config: WarehouseConfig) -> None:
    planner = PathPlanner(config.room_map)
    print(config.room_map.render_ascii(shelves=config.shelves))
    for product in config.products.all():
        shelf = config.shelves.get(product.shelf_id)
        path = planner.plan(config.room_map.robot_start, shelf.approach)
        print("\n{} ({}) -> shelf {}: {}".format(product.name, product.marker_id, shelf.shelf_id,
              " -> ".join("({:.2f},{:.2f})".format(x, y) for x, y in path) if path else "NO PATH"))


def synthetic_camera(config: WarehouseConfig, product_id: int) -> SimulatedWorldCamera:
    """Position-aware camera: what it sees depends on where the simulated robot is (bound in run_pick)."""
    if config.products.get_by_marker(product_id) is None:
        raise ConfigError("Unknown product {}".format(product_id))
    return SimulatedWorldCamera(build_world_markers(config, product_id))


def ask_product(config: WarehouseConfig):
    """Prompt until the user names a known product; None if they quit (q / empty EOF)."""
    print("\nAvailable products:")
    for p in config.products.all():
        shelf = config.shelves.get(p.shelf_id)
        print("  {}  {:<10} -> shelf {} (marker {})".format(
            p.marker_id, p.name, p.shelf_id, shelf.marker_id if shelf else "?"))
    while True:
        try:
            text = input("\nEnter a product ID or name (q to quit): ").strip()
        except EOFError:
            return None
        if text.lower() in ("q", "quit", "exit"):
            return None
        product = config.products.resolve(text) if text else None
        if product is not None:
            return product
        print("Unknown product '{}'. Try one of the IDs above.".format(text))


def run_interactive(args, config: WarehouseConfig) -> int:
    code = 0
    while True:
        product = ask_product(config)
        if product is None:
            return code
        code = run_pick(args, config, product)


def run_pick(args, config: WarehouseConfig, product=None) -> int:
    if product is None:
        product = config.products.resolve(args.pick)
    if product is None:
        print("Unknown product '{}'. Known: {}".format(
            args.pick, ", ".join("{} ({})".format(p.marker_id, p.name) for p in config.products.all())))
        return 2

    calibration = load_if_available(args.calibration_file)
    if calibration is None:
        log.info("CAMERA UNCALIBRATED: no valid %s. Marker pose (position/rotation) is unavailable; "
                 "run --calibrate for real measurements.", args.calibration_file)
    else:
        log.info("Camera calibrated (%dx%d, RMS %.2f px)", *calibration.image_size, calibration.rms_error_px)

    camera = synthetic_camera(config, product.marker_id) if args.synthetic \
        else WebcamSource(args.camera, args.width, args.height)

    def label(det):
        """Overlay text. A marker outside the configured ID ranges is not ours: say so rather than
        showing it as an unknown shelf, which reads like the target shelf was found."""
        if det.kind == "product":
            p = config.products.get_by_marker(det.marker_id)
            return "{} ({})".format(p.name if p else "unregistered product", det.marker_id)
        if det.kind == "shelf":
            shelf = config.shelves.get_by_marker(det.marker_id)
            return "Shelf {} ({})".format(shelf.shelf_id if shelf else "unregistered", det.marker_id)
        return "unknown marker {} [{}]".format(det.marker_id, det.dictionary)

    vision = Perception(camera, config, PoseEstimator(calibration),
                        display=not (args.no_display or args.synthetic), labeler=label)
    robot = SimulatedRobot(vision, config.room_map.robot_start, config.standoff_cm,
                           step_delay_s=0.0 if args.synthetic else 0.3)
    if args.synthetic:
        camera.bind(lambda: (*robot.current_position, robot.current_heading))
    # A live webcam is aimed by a person walking across a room; the synthetic camera is aimed by the
    # simulated robot. They need very different search budgets (see MachineSettings.for_live_camera).
    settings = MachineSettings(approach_tolerance_cm=config.approach_tolerance_cm) if args.synthetic \
        else MachineSettings.for_live_camera(approach_tolerance_cm=config.approach_tolerance_cm)
    machine = PickAndPlaceStateMachine(
        robot, config.products, config.shelves, PathPlanner(config.room_map),
        calibrated=calibration is not None, settings=settings,
        on_state=lambda s: vision.set_status(status_line(machine, s)))
    try:
        log.info("Request: pick %s (marker %d)", product.name, product.marker_id)
        result = machine.run(product.marker_id)
        return 0 if result.success else 1
    except UserAbort:
        log.info("Aborted by user.")
        return 130
    except KeyboardInterrupt:
        log.info("\nInterrupted.")
        return 130
    except CameraError as e:
        log.info("Camera error: %s", e)
        return 3
    except PlacementPreconditionError as e:
        # The placement guard fired: a real logic bug, not a runtime failure. Report it, do not crash.
        log.error("\nPLACEMENT REFUSED BY SAFETY GUARD: %s", e)
        return 4
    finally:
        vision.close()


def status_line(machine, state) -> str:
    """One line for the preview window: the state plus what the robot is holding and waiting for."""
    text = state.name
    carried = machine.carried_product
    if carried is not None:
        text += " | carrying {} ({})".format(carried.name, carried.marker_id)
    shelf = machine.target_shelf
    if shelf is not None and carried is not None:
        text += " -> show shelf {} marker {}".format(shelf.shelf_id, shelf.marker_id)
    # During the approach the distance is the thing the operator has to act on, so it goes on screen.
    if state == State.APPROACH_SHELF and machine.measured_distance_cm is not None:
        text += " | {:.0f} cm away, need {:.0f}".format(
            machine.measured_distance_cm, machine.robot.standoff_cm + machine.cfg.approach_tolerance_cm)
    return text


def run_calibrate(args) -> int:
    if not args.square_cm or args.square_cm <= 0:
        print("--square-cm is required: measure one printed square with a ruler. "
              "A wrong value silently scales every distance.")
        return 2
    try:
        cols, rows = (int(v) for v in args.board.lower().split("x"))
    except ValueError:
        print("--board must look like 7x6")
        return 2
    camera = WebcamSource(args.camera, args.width, args.height)
    try:
        return 0 if run_interactive_capture(camera, (cols, rows), args.square_cm, args.calibration_file) else 1
    except CalibrationError as e:
        print("Calibration failed: {}".format(e))
        return 1
    finally:
        camera.release()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args()
    try:
        if args.calibrate:
            return run_calibrate(args)
        config = load_config(args.config)
        if args.show_map:
            show_map(config)
            return 0
        if args.pick is None:
            return run_interactive(args, config)
        return run_pick(args, config)
    except (ConfigError, CameraError) as e:
        print("Error: {}".format(e))
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
