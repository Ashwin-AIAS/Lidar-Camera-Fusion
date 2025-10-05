# baby_kalman.py
# A tiny, step-by-step Kalman Filter demo for absolute beginners.
# State: [px, vx] -> position and velocity in 1D (so it's simpler).
# Sensors:
#  - IMU-like control: acceleration used in the prediction step
#  - GPS-like measurement: noisy position reading used in the update step
#
# Run: python baby_kalman.py

import numpy as np

def simulate_truth(total_time=8.0, dt=0.5):
    """Create a tiny ground-truth motion (1D)."""
    t = np.arange(0.0, total_time + 1e-9, dt)
    n = len(t)
    pos = np.zeros(n)
    vel = np.zeros(n)
    acc = np.zeros(n)
    # piecewise acceleration to make things interesting
    for k in range(1, n):
        if t[k] < 3.0:
            a = 0.8      # speeding up
        elif t[k] < 5.0:
            a = 0.0      # coast
        else:
            a = -0.6     # slowing down
        acc[k] = a
        vel[k] = vel[k-1] + a * dt
        pos[k] = pos[k-1] + vel[k-1] * dt + 0.5 * a * dt * dt
    return t, pos, vel, acc

def make_sensors(acc_true, pos_true, dt):
    """Create fake IMU (acc) and GPS (pos) measurements."""
    rng = np.random.RandomState(1)
    imu_noise = 0.05        # small accel noise
    gps_noise = 1.5         # larger position noise (meters)
    # IMU: acceleration measured each step (noise added)
    imu_meas = acc_true + rng.randn(len(acc_true)) * imu_noise
    # GPS: position measured each step but with bigger noise
    gps_meas = pos_true + rng.randn(len(pos_true)) * gps_noise
    return imu_meas, gps_meas

def baby_kalman(t, imu_meas, gps_meas, dt, print_every=1):
    """
    Very small Kalman filter for a 1D kinematic model:
      state x = [position, velocity]^T
    Motion model (continuous -> discrete with dt):
      px_k = px_{k-1} + vx_{k-1} * dt + 0.5 * a * dt^2
      vx_k = vx_{k-1} + a * dt
    We include acceleration 'a' as a control input (from IMU).
    """
    n = len(t)

    # State and covariance initialization
    x = np.array([0.0, 0.0])   # initial guess: at 0m, 0m/s
    P = np.eye(2) * 4.0        # initial uncertainty (we're quite unsure)

    # Matrices derived for this simple model
    # State-transition matrix (applies when we do "x = F @ x + B @ u")
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    # How acceleration affects the state
    B = np.array([0.5 * dt * dt, dt])

    # Measurement matrix: GPS measures position only (not velocity)
    H = np.array([1.0, 0.0]).reshape(1,2)

    # Noise covariances (tuning knobs!)
    accel_process_noise = 0.2   # we allow some process uncertainty from accel
    Q = np.array([[0.25*(dt**4), 0.5*(dt**3)],
                  [0.5*(dt**3),     dt**2]]) * (accel_process_noise**2)

    R = np.array([[1.5**2]])    # GPS measurement variance (we used gps_noise=1.5)

    # Storage for results (for possible plotting later)
    xs = np.zeros((n,2))
    Ps = np.zeros((n,2,2))

    print("Starting tiny Kalman demo (1D). I'll print prediction/update steps.\n")

    for k in range(n):
        a_meas = imu_meas[k]   # acceleration input (IMU)
        # --- PREDICTION STEP ---
        x_pred = F.dot(x) + B * a_meas
        P_pred = F.dot(P).dot(F.T) + Q

        if k % print_every == 0:
            print(f"Time {t[k]:4.2f}s - PREDICT")
            print(f"  Accel(measured) = {a_meas:.3f} m/s²")
            print(f"  Predicted state (pos, vel) = ({x_pred[0]:.3f}, {x_pred[1]:.3f})")
            print(f"  Predicted uncertainty P =\n{P_pred}\n")

        # --- UPDATE STEP (incorporate GPS position) ---
        z = np.array([gps_meas[k]])   # GPS position measurement (noisy)
        y = z - H.dot(x_pred)         # innovation (measurement minus predicted measurement)
        S = H.dot(P_pred).dot(H.T) + R
        K = P_pred.dot(H.T).dot(np.linalg.inv(S))   # Kalman gain (how much to trust the measurement)

        x = x_pred + (K.flatten() * y).reshape(2)   # updated state
        P = (np.eye(2) - K.dot(H)).dot(P_pred)      # updated covariance

        if k % print_every == 0:
            print(f"Time {t[k]:4.2f}s - UPDATE (GPS)")
            print(f"  GPS(measured pos) = {z[0]:.3f} m")
            print(f"  Innovation (z - Hx_pred) = {y[0]:.3f}")
            print(f"  Kalman Gain K = {K.flatten()}")
            print(f"  Updated state (pos, vel) = ({x[0]:.3f}, {x[1]:.3f})")
            print(f"  Updated uncertainty P =\n{P}\n")
            print("-"*60)

        xs[k] = x
        Ps[k] = P

    print("Done. The printed 'Updated state' is the filter's best guess each step.")
    return xs, Ps

def main():
    dt = 0.5
    t, pos_true, vel_true, acc_true = simulate_truth(total_time=8.0, dt=dt)
    imu_meas, gps_meas = make_sensors(acc_true, pos_true, dt)
    # run the tiny filter, print every step (print_every = 1)
    estimated_states, covariances = baby_kalman(t, imu_meas, gps_meas, dt, print_every=1)

    # After the run print a final comparison
    print("\nFinal comparison (true vs estimated):")
    print(f"{'time':>6} | {'true_pos':>8} {'est_pos':>9} | {'true_vel':>8} {'est_vel':>8}")
    print("-"*55)
    for k in range(len(t)):
        print(f"{t[k]:6.2f} | {pos_true[k]:8.3f} {estimated_states[k,0]:9.3f} | "
              f"{vel_true[k]:8.3f} {estimated_states[k,1]:8.3f}")

if __name__ == "__main__":
    main()
