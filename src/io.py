import datetime

def generate_mmddyyhrmin_string():
  """Generates a string representing the current date and time in MMDDYYHHMM format."""
  now = datetime.datetime.now()
  return now.strftime("%m%d%y%H%M")

