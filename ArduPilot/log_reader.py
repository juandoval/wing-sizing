# URL for online log plot: https://plot.ardupilot.org/#/

from ArdupilotLogReader import Ardupilot

log_file = '/ArduPilot/file_name.bin' # the log file, .bin

parser = Ardupilot(
    log_file, # the log file, .bin
    types = ['ARSP', 'ATT', 'BARO', 'GPS', 'IMU', 'RCIN', 'RCOU', 'BAT', 'MODE', 'NKF1', 'STAT', 'XKF1'],  # fields to read from the log
    )

print(parser.dfs) # a dict containing a dataframes of log data for each field requested.
print(parser.join_logs(['ARSP', 'ATT'])) #returns a pandas dataframe containing the ARSP and ATT data joined on time
print(parser.parms) # returns the parameters read from the top of the log
