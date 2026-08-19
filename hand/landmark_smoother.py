import math
import time


def _smoothing_factor(dt, cutoff):
    """
    Calculate exponential smoothing factor alpha:
    alpha = 1 / (1 + tau / dt) where tau = 1 / (2 * pi * cutoff)
    which simplifies to: (2 * pi * cutoff * dt) / (2 * pi * cutoff * dt + 1)
    """
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


def _exponential_smoothing(alpha, x, x_prev):
    """Standard low-pass exponential smoothing filter."""
    return alpha * x + (1.0 - alpha) * x_prev


class OneEuroFilter:
    """
    Enhanced One Euro Filter for 1D signal smoothing with velocity-adaptive
    jitter suppression and outlier jump protection.
    """

    def __init__(self, min_cutoff=0.8, beta=0.03, d_cutoff=1.0, max_jump=0.15):
        """
        Parameters:
            min_cutoff (float): Minimum cutoff frequency in Hz. Lower values provide
                                stronger stability during stationary / low-velocity states.
            beta (float): Speed coefficient. Higher values reduce lag during fast movements.
            d_cutoff (float): Cutoff frequency in Hz for filtering derivative/velocity.
            max_jump (float): Maximum allowed coordinate change per frame to prevent outlier jumps.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_jump = float(max_jump)

        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        """Reset internal filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x, timestamp=None):
        """
        Filter a 1D scalar value with adaptive speed control and jump limiting.

        Parameters:
            x (float): Current raw signal value.
            timestamp (float): Current timestamp in seconds (defaults to time.time()).

        Returns:
            float: Smoothed signal value.
        """
        if timestamp is None:
            timestamp = time.time()

        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = timestamp
            return x

        dt = timestamp - self.t_prev
        self.t_prev = timestamp

        # Guard against zero or negative delta time
        if dt <= 1e-6:
            return self.x_prev

        # If tracking was lost for over 0.4s, reset state to avoid jumping
        if dt > 0.4:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # 1. Landmark confidence protection: clamp sudden abnormal coordinate jumps
        diff = x - self.x_prev
        if abs(diff) > self.max_jump:
            # Smoothly interpolate rather than taking the full erratic jump
            x = self.x_prev + math.copysign(self.max_jump, diff)
            diff = x - self.x_prev

        # 2. Estimate rate of change (derivative) and filter it
        dx = diff / dt
        alpha_d = _smoothing_factor(dt, self.d_cutoff)
        dx_hat = _exponential_smoothing(alpha_d, dx, self.dx_prev)
        self.dx_prev = dx_hat

        # 3. Dynamic cutoff frequency based on velocity
        # Low velocity (stationary/jitter) -> cutoff ~ min_cutoff (strong smoothing)
        # High velocity (active motion)    -> cutoff increases (responsive/no lag)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 4. Filter the signal with the adaptive cutoff frequency
        alpha = _smoothing_factor(dt, cutoff)
        x_hat = _exponential_smoothing(alpha, x, self.x_prev)
        self.x_prev = x_hat

        return x_hat


class LandmarkSmoother:
    """
    Stabilizes 21 3D MediaPipe hand landmarks with hierarchical smoothing profiles:
    - Anchors (Wrist & MCP joints): Strongest smoothing to eliminate reference angle jitter.
    - Intermediate Joints (PIP/DIP): Balanced smoothing.
    - Fingertips: Responsive tracking with stationary jitter suppression.
    """

    # Landmark groups
    ANCHOR_INDICES      = {0, 5, 9, 13, 17}          # Wrist + MCP joints (highest importance)
    FINGERTIP_INDICES   = {4, 8, 12, 16, 20}         # Fingertips (responsive)
    INTERMEDIATE_INDICES = {1, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19}

    def __init__(
        self,
        min_cutoff=0.8,
        beta=0.03,
        d_cutoff=1.0,
        num_landmarks=21
    ):
        """
        Initialize landmark smoother with specialized filter profiles per landmark type.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.num_landmarks = num_landmarks

        self.filters = []
        for i in range(num_landmarks):
            if i in self.ANCHOR_INDICES:
                # Strong smoothing for wrist and palm MCP joints
                mc = 0.5
                b  = 0.02
                mj = 0.12
            elif i in self.FINGERTIP_INDICES:
                # Responsive tracking for fingertips with stationary stabilization
                mc = 0.9
                b  = 0.04
                mj = 0.18
            else:
                # Intermediate joints (PIP/DIP)
                mc = 0.7
                b  = 0.03
                mj = 0.15

            self.filters.append({
                "x": OneEuroFilter(min_cutoff=mc, beta=b, d_cutoff=d_cutoff, max_jump=mj),
                "y": OneEuroFilter(min_cutoff=mc, beta=b, d_cutoff=d_cutoff, max_jump=mj),
                "z": OneEuroFilter(min_cutoff=mc, beta=b, d_cutoff=d_cutoff, max_jump=mj),
            })

    def reset(self):
        """Reset all internal landmark filters."""
        for f in self.filters:
            f["x"].reset()
            f["y"].reset()
            f["z"].reset()

    def smooth(self, landmarks, timestamp=None):
        """
        Smooth a list of 21 landmark dictionaries.

        Parameters:
            landmarks (list of dict): List of 21 dicts [{'x': float, 'y': float, 'z': float}, ...]
            timestamp (float, optional): Timestamp in seconds.

        Returns:
            list of dict: Stabilized landmarks in the exact same format.
        """
        if not landmarks or len(landmarks) < self.num_landmarks:
            return landmarks

        if timestamp is None:
            timestamp = time.time()

        smoothed = []
        for i in range(self.num_landmarks):
            lm = landmarks[i]
            sx = self.filters[i]["x"].filter(float(lm["x"]), timestamp)
            sy = self.filters[i]["y"].filter(float(lm["y"]), timestamp)
            sz = self.filters[i]["z"].filter(float(lm["z"]), timestamp)

            smoothed.append({
                "x": sx,
                "y": sy,
                "z": sz
            })

        return smoothed
