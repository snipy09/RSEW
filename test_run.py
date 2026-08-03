from main import RegimeShiftWarningSystem
system = RegimeShiftWarningSystem(symbol='SPY', period_years=2, rolling_window=20, n_regimes=2, n_components=2)
res = system.run()
print(res)
