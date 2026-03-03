import csv
input_file = 'idk.csv'
output_file = 'fences.txt'

with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    points = [(row['lat'], row['lon']) for row in reader]

if points[0] != points[-1]:
    points.append(points[0])

with open(output_file, 'w') as f:
    for lat, lon in points:
        f.write(f"{lat},{lon}\n")