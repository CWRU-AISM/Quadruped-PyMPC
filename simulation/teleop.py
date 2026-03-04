"""
Real-time interactive teleop for Go2 using Quadruped-PyMPC.

Usage:
    conda activate quadruped_pympc_env
    cd Quadruped-PyMPC
    python simulation/teleop.py --gamepad
    python simulation/teleop.py --gamepad --scene perlin
    python simulation/teleop.py --gamepad --gait pace
"""
import os
import sys
import copy
import argparse
import threading
import time
import numpy as np
import mujoco

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr

from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco

# ── HORIPAD S mapping ─────────────────────────────────────────────────────────
AX_LX, AX_LY, AX_RX, AX_RY = 0, 1, 2, 3
BTN_Y, BTN_B, BTN_A, BTN_X = 0, 1, 2, 3
BTN_L, BTN_R = 4, 5
BTN_ZL, BTN_ZR = 6, 7
BTN_MINUS, BTN_PLUS = 8, 9
DEADZONE = 0.12


def apply_deadzone(v, dz=DEADZONE):
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


# ── Shared teleop state ──────────────────────────────────────────────────────
class TeleopState:
    def __init__(self, hip_height):
        self.hip_height = hip_height
        # Velocity commands (smoothed)
        self.vx_s = 0.0
        self.vy_s = 0.0
        self.wz_s = 0.0
        # Sit mode — full PD override of all legs
        self.is_sitting = False
        self.sit_blend = 0.0  # 0=standing, 1=fully sitting (smooth transition)
        # Front leg mode (L bumper held, or F toggle for keyboard)
        self.front_leg_mode = False
        self.front_leg_hip = 0.0    # J/L keys — hip abduction offset
        self.front_leg_thigh = 0.0  # I/K keys — thigh offset
        self.front_leg_calf = 0.0   # U/O keys — calf offset
        # Reset request
        self.reset_requested = False
        # Controller ref (set after construction)
        self.controller = None


# Go2 sitting joint targets (from build_scene.py keyframe, adapted for flat ground)
SIT_TARGETS = {
    "FL": np.array([0.0,  0.88, -0.94]),    # front: paws forward, moderate knee
    "FR": np.array([0.0,  0.88, -0.94]),
    "RL": np.array([0.0,  1.95, -2.723]),   # rear: thigh horizontal, calf tucked
    "RR": np.array([0.0,  1.95, -2.723]),
}
STAND_TARGETS = {
    "FL": np.array([0.0, 0.9, -1.8]),
    "FR": np.array([0.0, 0.9, -1.8]),
    "RL": np.array([0.0, 0.9, -1.8]),
    "RR": np.array([0.0, 0.9, -1.8]),
}


def gamepad_thread(env, cfg, state, controller):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[teleop] No gamepad found, using keyboard only.")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[teleop] Gamepad connected: {js.get_name()}")

    hip_h = cfg.hip_height
    max_lin = 2.0 * hip_h   # ~0.56 m/s — conservative
    max_ang = 1.5            # ~1.5 rad/s — conservative (stock uses 0.4)
    alpha = 0.05             # smooth acceleration

    while True:
        pygame.event.pump()

        # Check L bumper for front leg mode
        was_leg_mode = state.front_leg_mode
        state.front_leg_mode = js.get_button(BTN_L)

        if state.front_leg_mode:
            # Enter full_stance on first press to prevent MPC fight
            if not was_leg_mode:
                from quadruped_pympc.helpers.quadruped_utils import GaitType
                pgg = controller.wb_interface.pgg
                if pgg.gait_type != GaitType.FULL_STANCE.value:
                    state.vx_s = state.vy_s = state.wz_s = 0.0
                    if env._ref_base_lin_vel_H is not None:
                        env._ref_base_lin_vel_H[:] = 0.0
                    env._ref_base_ang_yaw_dot = 0.0
                    pgg.set_full_stance()
            # Right stick X = hip, right stick Y = thigh, ZL/ZR = calf
            state.front_leg_hip = apply_deadzone(js.get_axis(AX_RX))
            state.front_leg_thigh = apply_deadzone(js.get_axis(AX_RY))
            if js.get_button(BTN_ZL):
                state.front_leg_calf = max(state.front_leg_calf - 0.02, -1.0)
            elif js.get_button(BTN_ZR):
                state.front_leg_calf = min(state.front_leg_calf + 0.02, 1.0)
        else:
            if was_leg_mode and not state.is_sitting:
                # Restore gait when releasing L bumper
                controller.wb_interface.pgg.restore_previous_gait()
            state.front_leg_hip = 0.0
            state.front_leg_thigh = 0.0
            state.front_leg_calf = 0.0

        # Left stick: locomotion
        lx = apply_deadzone(js.get_axis(AX_LX))
        ly = apply_deadzone(js.get_axis(AX_LY))
        # Right stick X: yaw (only when not in front leg mode)
        rx = apply_deadzone(js.get_axis(AX_RX)) if not state.front_leg_mode else 0.0

        vx_t = -ly * max_lin
        vy_t = -lx * max_lin
        wz_t = -rx * max_ang

        # Speed modifiers
        if js.get_button(BTN_ZL):
            vx_t *= 0.3; vy_t *= 0.3; wz_t *= 0.3
        elif js.get_button(BTN_ZR):
            vx_t *= 2.0; vy_t *= 2.0

        # Smooth
        state.vx_s += alpha * (vx_t - state.vx_s)
        state.vy_s += alpha * (vy_t - state.vy_s)
        state.wz_s += alpha * (wz_t - state.wz_s)

        # Apply to env
        if env._ref_base_lin_vel_H is not None:
            env._ref_base_lin_vel_H[0] = np.clip(state.vx_s, -max_lin, max_lin)
            env._ref_base_lin_vel_H[1] = np.clip(state.vy_s, -max_lin, max_lin)
        env._ref_base_ang_yaw_dot = np.clip(state.wz_s, -max_ang, max_ang)

        # Auto-resume gait if stick is moved while in full stance
        if (abs(vx_t) > 0 or abs(vy_t) > 0 or abs(wz_t) > 0) and controller is not None:
            from quadruped_pympc.helpers.quadruped_utils import GaitType
            pgg = controller.wb_interface.pgg
            if pgg.gait_type == GaitType.FULL_STANCE.value:
                pgg.restore_previous_gait()
                print("  [GAIT RESUMED]")

        # A: zero velocities + toggle stand-still
        if js.get_button(BTN_A):
            state.vx_s = state.vy_s = state.wz_s = 0.0
            if env._ref_base_lin_vel_H is not None:
                env._ref_base_lin_vel_H[:] = 0.0
            env._ref_base_ang_yaw_dot = 0.0
            from quadruped_pympc.helpers.quadruped_utils import GaitType
            pgg = controller.wb_interface.pgg
            if pgg.gait_type != GaitType.FULL_STANCE.value:
                pgg.set_full_stance()
                print("  [STAND STILL]")
            else:
                pgg.restore_previous_gait()
                print("  [GAIT RESUMED]")
            time.sleep(0.3)

        # B: pause/resume
        if js.get_button(BTN_B):
            env.is_paused = not env.is_paused
            print(f"  [{'PAUSED' if env.is_paused else 'RESUMED'}]")
            time.sleep(0.3)

        # Y: sit/stand toggle (full PD override + full_stance)
        if js.get_button(BTN_Y):
            state.is_sitting = not state.is_sitting
            from quadruped_pympc.helpers.quadruped_utils import GaitType
            pgg = controller.wb_interface.pgg
            if state.is_sitting:
                state.vx_s = state.vy_s = state.wz_s = 0.0
                if env._ref_base_lin_vel_H is not None:
                    env._ref_base_lin_vel_H[:] = 0.0
                env._ref_base_ang_yaw_dot = 0.0
                pgg.set_full_stance()
                print("  [SITTING]")
            else:
                pgg.restore_previous_gait()
                print("  [STANDING]")
            time.sleep(0.3)

        # Minus: reset
        if js.get_button(BTN_MINUS):
            state.reset_requested = True
            time.sleep(0.3)

        time.sleep(0.01)


def _print_leg_status(teleop):
    print(f"  [LEGS] hip={teleop.front_leg_hip:+.1f}  thigh={teleop.front_leg_thigh:+.1f}  calf={teleop.front_leg_calf:+.1f}")


def make_key_callback(env, teleop, cfg):
    """Wrap env._key_callback with front-leg and extra controls."""
    # GLFW keycodes
    KEY_F = 70
    KEY_I = 73
    KEY_K = 75
    KEY_J = 74
    KEY_L = 76
    KEY_O = 79
    KEY_R = 82
    KEY_U = 85
    KEY_Y = 89
    KEY_LEFT_CTRL = 341
    KEY_RIGHT_CTRL = 345
    KEY_1 = 49  # trot
    KEY_2 = 50  # pace
    KEY_3 = 51  # crawl
    KEY_4 = 52  # bound

    GAIT_KEYS = {
        KEY_1: ('trot', 'TROT'),
        KEY_2: ('pace', 'PACE'),
        KEY_3: ('crawl', 'CRAWL'),
        KEY_4: ('bound', 'BOUND'),
    }

    def callback(keycode):
        # Either Ctrl: zero velocities + toggle stand-still
        if keycode in (KEY_LEFT_CTRL, KEY_RIGHT_CTRL):
            teleop.vx_s = teleop.vy_s = teleop.wz_s = 0.0
            if env._ref_base_lin_vel_H is not None:
                env._ref_base_lin_vel_H *= 0.0
            env._ref_base_ang_yaw_dot = 0.0
            if teleop.controller is not None:
                from quadruped_pympc.helpers.quadruped_utils import GaitType
                pgg = teleop.controller.wb_interface.pgg
                if pgg.gait_type != GaitType.FULL_STANCE.value:
                    pgg.set_full_stance()
                    print("  [STAND STILL]")
                else:
                    pgg.restore_previous_gait()
                    print("  [GAIT RESUMED]")
            return

        # 1-4: switch gait (transition through full_stance to avoid destabilizing)
        elif keycode in GAIT_KEYS and teleop.controller is not None:
            gait_name, label = GAIT_KEYS[keycode]
            gait_params = cfg.simulation_params['gait_params'][gait_name]
            pgg = teleop.controller.wb_interface.pgg
            # Brief full stance to plant all feet before switching
            pgg.set_full_stance()
            # Set new gait parameters
            pgg.duty_factor = gait_params['duty_factor']
            pgg.step_freq = gait_params['step_freq']
            pgg.previous_gait_type = gait_params['type']
            # Restore into new gait (calls reset → recalculates phase offsets)
            pgg.restore_previous_gait()
            cfg.simulation_params['gait'] = gait_name
            print(f"  [GAIT: {label}]")

        # Y: sit/stand toggle
        elif keycode == KEY_Y:
            teleop.is_sitting = not teleop.is_sitting
            if teleop.controller is not None:
                from quadruped_pympc.helpers.quadruped_utils import GaitType
                pgg = teleop.controller.wb_interface.pgg
                if teleop.is_sitting:
                    teleop.vx_s = teleop.vy_s = teleop.wz_s = 0.0
                    if env._ref_base_lin_vel_H is not None:
                        env._ref_base_lin_vel_H *= 0.0
                    env._ref_base_ang_yaw_dot = 0.0
                    pgg.set_full_stance()
                    print("  [SITTING]")
                else:
                    pgg.restore_previous_gait()
                    print("  [STANDING]")

        # F: toggle front leg mode (enters full_stance to prevent MPC fight)
        elif keycode == KEY_F:
            teleop.front_leg_mode = not teleop.front_leg_mode
            if teleop.controller is not None:
                from quadruped_pympc.helpers.quadruped_utils import GaitType
                pgg = teleop.controller.wb_interface.pgg
                if teleop.front_leg_mode:
                    teleop.vx_s = teleop.vy_s = teleop.wz_s = 0.0
                    if env._ref_base_lin_vel_H is not None:
                        env._ref_base_lin_vel_H *= 0.0
                    env._ref_base_ang_yaw_dot = 0.0
                    pgg.set_full_stance()
                else:
                    teleop.front_leg_hip = 0.0
                    teleop.front_leg_thigh = 0.0
                    teleop.front_leg_calf = 0.0
                    if not teleop.is_sitting:
                        pgg.restore_previous_gait()
            print(f"  [LEGS {'ON' if teleop.front_leg_mode else 'OFF'}]")

        # Front leg joint control (auto-enables leg mode + full_stance)
        # I/K = thigh raise/lower, U/O = calf extend/retract, J/L = hip abduction
        elif keycode in (KEY_I, KEY_K, KEY_U, KEY_O, KEY_J, KEY_L):
            if not teleop.front_leg_mode:
                teleop.front_leg_mode = True
                if teleop.controller is not None:
                    teleop.vx_s = teleop.vy_s = teleop.wz_s = 0.0
                    if env._ref_base_lin_vel_H is not None:
                        env._ref_base_lin_vel_H *= 0.0
                    env._ref_base_ang_yaw_dot = 0.0
                    teleop.controller.wb_interface.pgg.set_full_stance()
            if keycode == KEY_I:
                teleop.front_leg_thigh = min(teleop.front_leg_thigh + 0.3, 1.5)
            elif keycode == KEY_K:
                teleop.front_leg_thigh = max(teleop.front_leg_thigh - 0.3, -1.5)
            elif keycode == KEY_U:
                teleop.front_leg_calf = min(teleop.front_leg_calf + 0.3, 1.0)
            elif keycode == KEY_O:
                teleop.front_leg_calf = max(teleop.front_leg_calf - 0.3, -1.0)
            elif keycode == KEY_J:
                teleop.front_leg_hip = max(teleop.front_leg_hip - 0.3, -1.0)
            elif keycode == KEY_L:
                teleop.front_leg_hip = min(teleop.front_leg_hip + 0.3, 1.0)
            _print_leg_status(teleop)

        elif keycode == KEY_R:
            teleop.reset_requested = True
        else:
            env._key_callback(keycode)

    return callback


def compute_pose_override(env, state, legs_order, dt):
    """Override leg torques for sit mode and/or front leg mode.

    Returns dict of {leg_name: torque_array} for legs that should be overridden,
    or None if no override needed.
    """
    # Update sit blend (smooth transition over ~1s)
    blend_rate = dt * 2.0  # reach full sit in ~0.5s
    if state.is_sitting:
        state.sit_blend = min(state.sit_blend + blend_rate, 1.0)
    else:
        state.sit_blend = max(state.sit_blend - blend_rate, 0.0)

    overrides = {}
    kp, kd = 40.0, 4.0

    # Sit mode: PD control ALL legs toward sit pose
    if state.sit_blend > 0.01:
        for leg in legs_order:
            idx = env.legs_qpos_idx[leg]
            q = env.mjData.qpos[idx]
            dq = env.mjData.qvel[env.legs_qvel_idx[leg]]

            # Blend between standing and sitting targets
            targets = (1.0 - state.sit_blend) * STAND_TARGETS[leg] + \
                      state.sit_blend * SIT_TARGETS[leg]
            torque = kp * (targets - q) + kd * (0.0 - dq)
            overrides[leg] = torque

    # Front leg mode: override front legs with manual control
    # Base targets match current pose (sit vs stand), offsets add on top
    if state.front_leg_mode:
        if state.is_sitting:
            base = SIT_TARGETS["FL"]  # [0.0, 0.88, -0.94]
        else:
            base = STAND_TARGETS["FL"]  # [0.0, 0.9, -1.8]
        hip_target = base[0] + state.front_leg_hip * 0.4
        thigh_target = base[1] + state.front_leg_thigh * 0.8
        calf_target = base[2] + state.front_leg_calf * 0.6
        targets = np.array([hip_target, thigh_target, calf_target])

        for leg in ["FL", "FR"]:
            idx = env.legs_qpos_idx[leg]
            q = env.mjData.qpos[idx]
            dq = env.mjData.qvel[env.legs_qvel_idx[leg]]
            torque = kp * (targets - q) + kd * (0.0 - dq)
            overrides[leg] = torque

    return overrides if overrides else None


def main():
    parser = argparse.ArgumentParser(description="Quadruped-PyMPC Teleop")
    parser.add_argument("--gamepad", action="store_true")
    parser.add_argument("--scene", choices=["flat", "random_boxes", "random_pyramids", "perlin", "fetch", "unitree"], default="flat")
    parser.add_argument("--gait", choices=["trot", "pace", "crawl", "bound"], default="trot")
    parser.add_argument("--mpc", choices=["nominal", "sampling", "input_rates"], default="nominal")
    args = parser.parse_args()

    from quadruped_pympc import config as cfg
    cfg.simulation_params["scene"] = args.scene
    cfg.simulation_params["gait"] = args.gait
    cfg.simulation_params["mode"] = "human"
    cfg.mpc_params["type"] = args.mpc

    # ── Create environment ────────────────────────────────────────────────
    robot_name = cfg.robot
    hip_height = cfg.hip_height
    simulation_dt = cfg.simulation_params["dt"]

    env = QuadrupedEnv(
        robot=robot_name,
        scene=cfg.simulation_params["scene"],
        sim_dt=simulation_dt,
        ref_base_lin_vel=0.0,
        ref_base_ang_vel=0.0,
        ground_friction_coeff=0.8,
        base_vel_command_type="human",
        state_obs_names=(),
    )
    env.mjModel.opt.gravity[2] = -cfg.gravity_constant

    if cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))

    env.reset(random=False)

    # ── Set initial position for custom scenes ────────────────────────────
    if args.scene == "fetch":
        # Place Go2 on top of the platform (top z=0.30, standing height ~0.28)
        env.mjData.qpos[0:3] = [0.70, 0.0, 0.58]
        mujoco.mj_forward(env.mjModel, env.mjData)

    # ── Teleop state (created before viewer so key callback can reference it)
    teleop = TeleopState(hip_height)

    # Create viewer with full UI panels + custom key callback
    env.viewer = mujoco.viewer.launch_passive(
        env.mjModel,
        env.mjData,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=make_key_callback(env, teleop, cfg),
    )
    env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    mujoco.mjv_defaultFreeCamera(env.mjModel, env.viewer.cam)

    # ── Torque limits ─────────────────────────────────────────────────────
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft,
    )

    legs_order = ["FL", "FR", "RL", "RR"]
    feet_traj_geom_ids = None
    feet_GRF_geom_ids = LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)

    # ── MPC controller ────────────────────────────────────────────────────
    controller = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=(
            "ref_base_height", "ref_base_angles", "ref_feet_pos",
            "nmpc_GRFs", "nmpc_footholds", "swing_time",
            "phase_signal", "lift_off_positions",
        ),
    )

    teleop.controller = controller

    # ── Start gamepad ─────────────────────────────────────────────────────
    if args.gamepad:
        threading.Thread(target=gamepad_thread, args=(env, cfg, teleop, controller), daemon=True).start()

    # ── Print controls ────────────────────────────────────────────────────
    print("\n=== Quadruped-PyMPC Teleop ===")
    if args.gamepad:
        print("  Left stick       : forward/back + strafe")
        print("  Right stick X    : turn left/right")
        print("  ZL               : creep (0.3x)")
        print("  ZR               : sprint (2x)")
        print("  A                : stand still / resume gait")
        print("  B                : pause/resume")
        print("  Y                : sit / stand up")
        print("  L (hold)         : front leg mode (R-stick: hip/thigh, ZL/ZR: calf)")
        print("  Minus            : reset simulation")
    print("  Arrow Up/Down    : forward/back")
    print("  Arrow Left/Right : turn")
    print("  Ctrl (L or R)    : stand still / resume gait")
    print("  Space            : pause/resume")
    print("  Y                : sit / stand up")
    print("  1/2/3/4          : gait — trot / pace / crawl / bound")
    print("  F                : toggle front leg mode (F again to release)")
    print("  I/K              : thigh raise / lower")
    print("  U/O              : calf extend / retract")
    print("  J/L              : hip abduct left / right")
    print("  R                : reset simulation")
    print()

    # ── Real-time control loop ────────────────────────────────────────────
    RENDER_FREQ = 30
    last_render_time = time.time()
    step_count = 0

    while True:
        step_start = time.perf_counter()

        # Handle reset
        if teleop.reset_requested:
            teleop.reset_requested = False
            teleop.vx_s = teleop.vy_s = teleop.wz_s = 0.0
            teleop.is_sitting = False
            cfg.simulation_params['ref_z'] = hip_height
            env.reset(random=False)
            if args.scene == "fetch":
                env.mjData.qpos[0:3] = [0.70, 0.0, 0.58]
                mujoco.mj_forward(env.mjModel, env.mjData)
            controller.reset(initial_feet_pos=env.feet_pos(frame="world"))
            print("  [RESET]")
            continue

        # Read state
        feet_pos = env.feet_pos(frame="world")
        feet_vel = env.feet_vel(frame="world")
        hip_pos = env.hip_positions(frame="world")
        base_lin_vel = env.base_lin_vel(frame="world")
        base_ang_vel = env.base_ang_vel(frame="base")
        base_ori_euler_xyz = env.base_ori_euler_xyz
        base_pos = copy.deepcopy(env.base_pos)
        com_pos = copy.deepcopy(env.com)

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        if cfg.simulation_params["use_inertia_recomputation"]:
            inertia = env.get_base_inertia().flatten()
        else:
            inertia = cfg.inertia.flatten()

        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        legs_qvel_idx = env.legs_qvel_idx
        legs_qpos_idx = env.legs_qpos_idx
        joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR,
                              RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)

        legs_mass_matrix = env.legs_mass_matrix
        legs_qfrc_bias = env.legs_qfrc_bias
        legs_qfrc_passive = env.legs_qfrc_passive
        feet_jac = env.feet_jacobians(frame="world", return_rot_jac=False)
        feet_jac_dot = env.feet_jacobians_dot(frame="world", return_rot_jac=False)

        # Compute MPC action
        tau = controller.compute_actions(
            com_pos, base_pos, base_lin_vel, base_ori_euler_xyz, base_ang_vel,
            feet_pos, hip_pos, joints_pos, None, legs_order, simulation_dt,
            ref_base_lin_vel, ref_base_ang_vel, env.step_num,
            qpos, qvel, feet_jac, feet_jac_dot, feet_vel,
            legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix,
            legs_qpos_idx, legs_qvel_idx, tau, inertia, env.mjData.contact,
        )

        # Override legs for sit mode / front leg mode
        pose_overrides = compute_pose_override(env, teleop, legs_order, simulation_dt)
        if pose_overrides is not None:
            for leg, torque in pose_overrides.items():
                tau[leg] = torque

        # Clip torques
        for leg in legs_order:
            tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
            tau[leg] = np.clip(tau[leg], tau_min, tau_max)

        # Apply action
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL
        action[env.legs_tau_idx.FR] = tau.FR
        action[env.legs_tau_idx.RL] = tau.RL
        action[env.legs_tau_idx.RR] = tau.RR

        state, reward, is_terminated, is_truncated, info = env.step(action=action)

        # Render at 30Hz
        now = time.time()
        if now - last_render_time > 1.0 / RENDER_FREQ:
            ctrl_state = controller.get_obs()

            # Swing trajectory visualization
            feet_traj_geom_ids = plot_swing_mujoco(
                viewer=env.viewer,
                swing_traj_controller=controller.wb_interface.stc,
                swing_period=controller.wb_interface.stc.swing_period,
                swing_time=LegsAttr(
                    FL=ctrl_state["swing_time"][0],
                    FR=ctrl_state["swing_time"][1],
                    RL=ctrl_state["swing_time"][2],
                    RR=ctrl_state["swing_time"][3],
                ),
                lift_off_positions=ctrl_state["lift_off_positions"],
                nmpc_footholds=ctrl_state["nmpc_footholds"],
                ref_feet_pos=ctrl_state["ref_feet_pos"],
                early_stance_detector=controller.wb_interface.esd,
                geom_ids=feet_traj_geom_ids,
            )

            # GRF arrows
            _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)
            for leg_name in legs_order:
                feet_GRF_geom_ids[leg_name] = render_vector(
                    env.viewer, vector=feet_GRF[leg_name],
                    pos=feet_pos[leg_name],
                    scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                    color=np.array([0, 1, 0, 0.5]),
                    geom_id=feet_GRF_geom_ids[leg_name],
                )

            env.render()
            last_render_time = now

        # HUD every 0.5s
        step_count += 1
        if step_count % int(0.5 / simulation_dt) == 0:
            vx = ref_base_lin_vel[0] if ref_base_lin_vel is not None else 0
            wz = ref_base_ang_vel[2] if ref_base_ang_vel is not None else 0
            mode = "SIT" if teleop.is_sitting else ("LEGS" if teleop.front_leg_mode else "WALK")
            print(f"\r  [{mode}] vx={vx:+.2f}  wz={wz:+.2f}  z={base_pos[2]:.3f}  t={env.simulation_time:.1f}s  ", end="", flush=True)

        # Real-time pacing
        elapsed = time.perf_counter() - step_start
        sleep_t = simulation_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

        # If viewer closed, exit
        if env.viewer is not None and not env.viewer.is_running():
            break

    env.close()
    print("\n[teleop] Done.")


if __name__ == "__main__":
    main()
