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
        # Sit mode
        self.is_sitting = False
        # Front leg mode (L bumper held, or F toggle for keyboard)
        self.front_leg_mode = False
        self.front_leg_rx = 0.0  # right stick X / J/L keys (hip abduction)
        self.front_leg_ry = 0.0  # right stick Y / I/K keys (thigh)
        # Reset request
        self.reset_requested = False


def gamepad_thread(env, cfg, state):
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
        state.front_leg_mode = js.get_button(BTN_L)

        if state.front_leg_mode:
            # In front leg mode, right stick controls front legs
            state.front_leg_rx = apply_deadzone(js.get_axis(AX_RX))
            state.front_leg_ry = apply_deadzone(js.get_axis(AX_RY))
        else:
            state.front_leg_rx = 0.0
            state.front_leg_ry = 0.0

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

        # A: zero all velocities (e-stop)
        if js.get_button(BTN_A):
            state.vx_s = state.vy_s = state.wz_s = 0.0
            if env._ref_base_lin_vel_H is not None:
                env._ref_base_lin_vel_H[:] = 0.0
            env._ref_base_ang_yaw_dot = 0.0

        # B: pause/resume
        if js.get_button(BTN_B):
            env.is_paused = not env.is_paused
            print(f"  [{'PAUSED' if env.is_paused else 'RESUMED'}]")
            time.sleep(0.3)

        # Y: sit/stand toggle
        if js.get_button(BTN_Y):
            state.is_sitting = not state.is_sitting
            if state.is_sitting:
                cfg.simulation_params['ref_z'] = hip_h * 0.5
                print("  [SIT]")
            else:
                cfg.simulation_params['ref_z'] = hip_h
                print("  [STAND]")
            time.sleep(0.3)

        # Minus: reset
        if js.get_button(BTN_MINUS):
            state.reset_requested = True
            time.sleep(0.3)

        time.sleep(0.01)


def make_key_callback(env, teleop, cfg):
    """Wrap env._key_callback with front-leg and extra controls."""
    # GLFW keycodes
    KEY_F = 70
    KEY_I = 73
    KEY_K = 75
    KEY_J = 74
    KEY_L = 76
    KEY_R = 82

    def callback(keycode):
        if keycode == KEY_F:
            teleop.front_leg_mode = not teleop.front_leg_mode
            if not teleop.front_leg_mode:
                teleop.front_leg_rx = 0.0
                teleop.front_leg_ry = 0.0
            print(f"  [LEGS {'ON' if teleop.front_leg_mode else 'OFF'}]")
        elif keycode == KEY_I:
            teleop.front_leg_ry = min(teleop.front_leg_ry + 0.25, 1.0)
        elif keycode == KEY_K:
            teleop.front_leg_ry = max(teleop.front_leg_ry - 0.25, -1.0)
        elif keycode == KEY_J:
            teleop.front_leg_rx = max(teleop.front_leg_rx - 0.25, -1.0)
        elif keycode == KEY_L:
            teleop.front_leg_rx = min(teleop.front_leg_rx + 0.25, 1.0)
        elif keycode == KEY_R:
            teleop.reset_requested = True
        else:
            env._key_callback(keycode)

    return callback


def compute_front_leg_pd(env, state, legs_order):
    """Override front leg torques with PD control when L bumper is held."""
    if not state.front_leg_mode:
        return None

    kp, kd = 40.0, 4.0

    # Standing pose targets
    hip_target = 0.0
    thigh_target = 0.9
    calf_target = -1.8

    # Modulate with right stick
    thigh_target += state.front_leg_ry * 0.5   # forward/back
    hip_target += state.front_leg_rx * 0.3     # abduction

    front_torques = {}
    for leg in ["FL", "FR"]:
        idx = env.legs_qpos_idx[leg]
        q = env.mjData.qpos[idx]
        dq = env.mjData.qvel[env.legs_qvel_idx[leg]]

        targets = np.array([hip_target, thigh_target, calf_target])
        torque = kp * (targets - q) + kd * (0.0 - dq)
        front_torques[leg] = torque

    return front_torques


def main():
    parser = argparse.ArgumentParser(description="Quadruped-PyMPC Teleop")
    parser.add_argument("--gamepad", action="store_true")
    parser.add_argument("--scene", choices=["flat", "random_boxes", "random_pyramids", "perlin"], default="flat")
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

    # ── Start gamepad ─────────────────────────────────────────────────────
    if args.gamepad:
        threading.Thread(target=gamepad_thread, args=(env, cfg, teleop), daemon=True).start()

    # ── Print controls ────────────────────────────────────────────────────
    print("\n=== Quadruped-PyMPC Teleop ===")
    if args.gamepad:
        print("  Left stick       : forward/back + strafe")
        print("  Right stick X    : turn left/right")
        print("  ZL               : creep (0.3x)")
        print("  ZR               : sprint (2x)")
        print("  A                : e-stop (zero velocities)")
        print("  B                : pause/resume")
        print("  Y                : sit/stand toggle")
        print("  L (hold)         : front leg mode (right stick controls legs)")
        print("  Minus            : reset simulation")
    print("  Arrow Up/Down    : forward/back")
    print("  Arrow Left/Right : turn")
    print("  Ctrl             : stop all motion")
    print("  Space            : pause/resume")
    print("  F                : toggle front leg mode")
    print("  I/K              : front legs forward/back")
    print("  J/L              : front legs abduct left/right")
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

        # Override front legs if in front leg mode
        front_pd = compute_front_leg_pd(env, teleop, legs_order)
        if front_pd is not None:
            for leg in ["FL", "FR"]:
                tau[leg] = front_pd[leg]

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
