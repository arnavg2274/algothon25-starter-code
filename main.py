import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

# Advanced strategy parameters based on professional research
LOOKBACK_DAYS = 20  # Shorter for more responsive signals
TREND_LOOKBACK = 100  # Long-term trend identification
MAX_POSITION_PCT = 0.4  # Conservative position sizing for better risk-adjusted returns
VOLATILITY_THRESHOLD = 0.012  # Even stricter volatility filter
ENTRY_THRESHOLD = 2.5  # Higher threshold for ultra-selective entry
EXIT_THRESHOLD = 0.8  # Quicker exits as recommended by research
MIN_PRICE = 10.0  # Higher quality stocks only
MAX_HOLDING_DAYS = 5  # Maximum holding period as per research
RSI_OVERSOLD = 20  # RSI-based mean reversion levels
RSI_OVERBOUGHT = 80

# Position tracking for holding period management
position_entry_day = np.full(nInst, -1)
global_day_counter = 0

def calculate_rsi(prices, period=2):
    """Calculate 2-period RSI as recommended in research"""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def identify_trend(prices, short_period=20, long_period=100):
    """Identify primary trend direction using dual moving averages"""
    if len(prices) < long_period:
        return 0  # Neutral
    
    short_ma = np.mean(prices[-short_period:])
    long_ma = np.mean(prices[-long_period:])
    
    if short_ma > long_ma * 1.02:  # 2% buffer for strong uptrend
        return 1
    elif short_ma < long_ma * 0.98:  # 2% buffer for strong downtrend
        return -1
    else:
        return 0  # Sideways/unclear trend

def getMyPosition(prcSoFar):
    global currentPos, position_entry_day, global_day_counter
    (nins, nt) = prcSoFar.shape
    global_day_counter = nt
    
    # Need sufficient history for all calculations
    if nt < TREND_LOOKBACK + 10:
        return np.zeros(nins)
    
    # Calculate current prices and position limits
    currentPrices = prcSoFar[:, -1]
    posLimits = np.array([int(x) for x in 10000 / np.maximum(currentPrices, MIN_PRICE)])
    
    # New position array
    newPos = np.zeros(nins)
    
    for i in range(nins):
        # Skip low-quality stocks
        if currentPrices[i] < MIN_PRICE:
            newPos[i] = 0
            position_entry_day[i] = -1
            continue
            
        # Get price history for this instrument
        price_history = prcSoFar[i, :]
        
        # Calculate trend direction
        trend = identify_trend(price_history)
        
        # Skip if no clear trend (reduces trading frequency and costs)
        if trend == 0:
            # Gradually exit positions in unclear trends
            newPos[i] = int(currentPos[i] * 0.5)
            continue
        
        # Calculate volatility over recent period
        if nt >= LOOKBACK_DAYS + 1:
            returns = np.log(price_history[-LOOKBACK_DAYS:] / price_history[-LOOKBACK_DAYS-1:-1])
            volatility = np.std(returns)
        else:
            volatility = 0.02  # Default to high volatility
            
        # Skip volatile instruments
        if volatility > VOLATILITY_THRESHOLD:
            newPos[i] = int(currentPos[i] * 0.7)  # Gradual exit
            continue
            
        # Calculate mean reversion signals
        mean_returns = np.mean(returns) if nt >= LOOKBACK_DAYS + 1 else 0
        current_return = np.log(currentPrices[i] / prcSoFar[i, -2])
        
        # Z-score for mean reversion
        z_score = (current_return - mean_returns) / volatility if volatility > 0 else 0
        
        # Calculate RSI for additional confirmation
        rsi = calculate_rsi(price_history[-10:]) if len(price_history) >= 10 else 50
        
        # Check if we should exit due to holding period
        if position_entry_day[i] >= 0 and (global_day_counter - position_entry_day[i]) >= MAX_HOLDING_DAYS:
            newPos[i] = 0  # Force exit after max holding period
            position_entry_day[i] = -1
            continue
            
        # PROFESSIONAL MEAN REVERSION LOGIC (trend-aligned)
        current_position = currentPos[i]
        
        if trend == 1:  # UPTREND - Look for dips to buy
            if (z_score < -ENTRY_THRESHOLD and rsi < RSI_OVERSOLD and current_position == 0):
                # Strong oversold signal in uptrend - BUY
                signal_strength = min(1.0, abs(z_score) / 4.0)
                newPos[i] = int(MAX_POSITION_PCT * posLimits[i] * signal_strength)
                position_entry_day[i] = global_day_counter
            elif current_position > 0 and (z_score > -EXIT_THRESHOLD or rsi > 70):
                # Exit long position - price has reverted or become overbought
                newPos[i] = 0
                position_entry_day[i] = -1
            else:
                newPos[i] = current_position  # Hold
                
        elif trend == -1:  # DOWNTREND - Look for rallies to short
            if (z_score > ENTRY_THRESHOLD and rsi > RSI_OVERBOUGHT and current_position == 0):
                # Strong overbought signal in downtrend - SHORT
                signal_strength = min(1.0, abs(z_score) / 4.0)
                newPos[i] = -int(MAX_POSITION_PCT * posLimits[i] * signal_strength)
                position_entry_day[i] = global_day_counter
            elif current_position < 0 and (z_score < EXIT_THRESHOLD or rsi < 30):
                # Exit short position - price has reverted or become oversold
                newPos[i] = 0
                position_entry_day[i] = -1
            else:
                newPos[i] = current_position  # Hold
        
        # Apply position limits
        newPos[i] = np.clip(newPos[i], -posLimits[i], posLimits[i])
    
    # Update global position tracking
    currentPos = newPos.copy()
    
    return newPos