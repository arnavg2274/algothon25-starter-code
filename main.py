import numpy as np

##### MOMENTUM + MEAN REVERSION + BREAKOUT + VOLATILITY SIZING - VERSION 5 #####
### Added volatility-based position sizing for better risk management ###

nInst = 50
currentPos = np.zeros(nInst)

# Strategy parameters
MOMENTUM_DAYS = 50  # Momentum lookback period
MEAN_REVERT_DAYS = 5  # Short-term mean reversion period  
VOLATILITY_DAYS = 10  # Volatility calculation period
BREAKOUT_DAYS = 25  # Breakout lookback period
BASE_POSITION_SIZE = 1.15  # Base position size (increased slightly)
MIN_PRICE = 1.0  # Avoid penny stocks
MAX_VOLATILITY = 0.225  # Tighter volatility filter
TARGET_VOLATILITY = 0.02  # Target volatility for position sizing
MOMENTUM_THRESHOLD = 0.02  # 2% momentum threshold
MEAN_REVERT_THRESHOLD = 0.0350  # 1.5% mean reversion threshold
BREAKOUT_THRESHOLD = 0.00005  # 0.5% above/below recent high/low


def calculate_volatility(prices, days=10):
    """Calculate daily volatility over specified period"""
    if len(prices) < days + 1:
        return 0.05  # Default to high volatility if not enough data
    
    # Calculate daily returns
    returns = np.diff(prices[-days-1:]) / prices[-days-1:-1]
    return np.std(returns)

def calculate_volatility_adjusted_size(base_size, volatility, target_vol=0.02):
    """Calculate position size adjusted for volatility"""
    if volatility <= 0:
        return base_size
    
    # Size inversely to volatility: lower vol = larger position, higher vol = smaller position
    vol_ratio = target_vol / volatility
    # Cap the adjustment between 0.3x and 2.0x to avoid extreme positions
    vol_ratio = np.clip(vol_ratio, 0.3, 2.0)
    
    return base_size * vol_ratio

def calculate_mean_reversion_signal(prices, days=5):
    """Calculate mean reversion signal - recent price vs short-term average"""
    if len(prices) < days + 1:
        return 0.0
    
    recent_avg = np.mean(prices[-days-1:-1])  # Average of last N days (excluding today)
    current_price = prices[-1]
    
    # Negative signal means price is below average (buy signal)
    # Positive signal means price is above average (sell signal)
    return (current_price - recent_avg) / recent_avg

def calculate_breakout_signal(prices, days=20):
    """Calculate breakout signal - current price vs recent high/low"""
    if len(prices) < days + 1:
        return 0.0
    
    # Get recent high and low (excluding today)
    recent_high = np.max(prices[-days-1:-1])
    recent_low = np.min(prices[-days-1:-1])
    current_price = prices[-1]
    
    # Check for breakout above recent high
    if current_price > recent_high * (1 + BREAKOUT_THRESHOLD):
        return 1.0  # Bullish breakout
    
    # Check for breakdown below recent low
    elif current_price < recent_low * (1 - BREAKOUT_THRESHOLD):
        return -1.0  # Bearish breakout
    
    return 0.0  # No breakout

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    
    # Need enough history for all calculations
    if nt < max(MOMENTUM_DAYS, MEAN_REVERT_DAYS, VOLATILITY_DAYS, BREAKOUT_DAYS) + 1:
        return np.zeros(nins)
    
    currentPrices = prcSoFar[:, -1]
    posLimits = np.array([int(x) for x in 10000 / np.maximum(currentPrices, MIN_PRICE)])
    
    newPos = np.zeros(nins)
    
    for i in range(nins):
        # Skip low-quality stocks
        if currentPrices[i] < MIN_PRICE:
            continue
        
        # Calculate volatility for filtering AND position sizing
        volatility = calculate_volatility(prcSoFar[i, :], VOLATILITY_DAYS)
        if volatility > MAX_VOLATILITY:
            continue  # Skip volatile stocks
            
        # Calculate volatility-adjusted position size
        adjusted_position_size = calculate_volatility_adjusted_size(
            BASE_POSITION_SIZE, volatility, TARGET_VOLATILITY
        )
        
        # Calculate all signals
        price_now = currentPrices[i]
        price_past = prcSoFar[i, -MOMENTUM_DAYS-1]
        momentum = (price_now - price_past) / price_past
        
        mean_revert_signal = calculate_mean_reversion_signal(prcSoFar[i, :], MEAN_REVERT_DAYS)
        breakout_signal = calculate_breakout_signal(prcSoFar[i, :], BREAKOUT_DAYS)
        
        # Multi-signal strategy with priority system
        signal_strength = 0.0
        
        # Priority 1: Breakout signals (strongest - catch big moves early)
        if breakout_signal != 0:
            signal_strength = breakout_signal * 1.2  # Amplify breakout signals
        
        # Priority 2: Strong momentum signals (trend-following)
        elif momentum > MOMENTUM_THRESHOLD:
            signal_strength = 1.0  # Buy on strong uptrend
        elif momentum < -MOMENTUM_THRESHOLD:
            signal_strength = -1.0  # Sell on strong downtrend
        
        # Priority 3: Mean reversion signals (contrarian) - only when momentum is weak
        elif abs(momentum) < MOMENTUM_THRESHOLD * 0.5:  # Weak momentum
            if mean_revert_signal < -MEAN_REVERT_THRESHOLD:
                signal_strength = 0.8  # Buy dip (price below recent average)
            elif mean_revert_signal > MEAN_REVERT_THRESHOLD:
                signal_strength = -0.8  # Sell rally (price above recent average)
        
        # Apply volatility-adjusted position sizing
        if signal_strength != 0:
            position = int(adjusted_position_size * posLimits[i] * signal_strength)
            newPos[i] = position
    
    # Apply position limits
    newPos = np.clip(newPos, -posLimits, posLimits)
    
    # Update tracking
    currentPos = newPos.copy()
    
    return newPos.astype(int)