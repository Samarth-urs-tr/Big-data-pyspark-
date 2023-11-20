from pyspark import SparkContext, SparkConf
import sys

# Initialize Spark context
conf = SparkConf().setAppName("taxi_analysis")
sc = SparkContext.getOrCreate()

# Define the file path
file_path = sys.argv[1]

raw_data = sc.textFile(file_path)

split_data = raw_data.map(lambda line: line.split(','))

# Get 'medallion' and 'hack_license' columns
medallion_hack_license = split_data.map(lambda row: (row[0], row[1]))

# Part 1: Find the top ten taxis with the largest number of drivers
taxi_drivers_count = medallion_hack_license.distinct().groupByKey().mapValues(len)
top_ten_taxis = taxi_drivers_count.takeOrdered(10, key=lambda x: -x[1])

print("Top Ten Taxis with the Largest Number of Drivers:")
for taxi in top_ten_taxis:
    print(taxi)

# Task 2: Analyze the top 10 best drivers in terms of average money per minute
filtered_data = split_data.filter(lambda row: float(row[16]) > 0 and float(row[4]) > 60)

money_per_minute_data = filtered_data.map(lambda row: (row[1], float(row[16]) / (float(row[4]) / 60)))

driver_avg_money_per_minute = money_per_minute_data.groupByKey().mapValues(lambda values: sum(values) / len(values))

# Find the top 10 best drivers in terms of average money per minute
top_10_drivers = driver_avg_money_per_minute.takeOrdered(10, key=lambda x: -x[1])

print("Top 10 Best Drivers (driver, money per minute):")
for driver in top_10_drivers:
    print(driver)

# Task 3: Calculate profit ratio (surcharge / trip_distance) and find the best hour
# Analyzing the top 10 best drivers in terms of average money per minute
columns = ["medallion", "hack_license", "pickup_datetime", "dropoff_datetime", "trip_time_in_secs",
           "trip_distance", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
           "payment_type", "fare_amount", "surcharge", "mta_tax", "tip_amount", "tolls_amount", "total_amount"]

taxilinesWithColumns = split_data.map(lambda p: dict(zip(columns, p)))

# Mapping (hour_of_day, (surcharge, trip_distance))
hourly_data = taxilinesWithColumns.map(lambda p: (int(p["pickup_datetime"].split()[1].split(':')[0]), (float(p["surcharge"]), float(p["trip_distance"]))))

filtered_hourly_data = hourly_data.filter(lambda p: p[1][0] > 0)


profit_ratio_data = filtered_hourly_data.map(lambda p: (p[0], p[1][0] / p[1][1] if p[1][1] != 0 else 0))
hourly_average_profit_ratio = profit_ratio_data.groupByKey().mapValues(lambda values: sum(values) / len(values) if len(values) > 0 else 0)
filtered_hourly_average_profit_ratio = hourly_average_profit_ratio.filter(lambda x: x[1] > 0)
if filtered_hourly_average_profit_ratio.isEmpty():
    print("No valid records found.")
else:
    best_hour = filtered_hourly_average_profit_ratio.max(lambda x: x[1])
    print("The best time of the day with the highest profit ratio is hour", best_hour[0])




sc.stop()
