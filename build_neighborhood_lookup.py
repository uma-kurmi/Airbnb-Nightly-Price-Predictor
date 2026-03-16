import os
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("Build Neighborhood Lookup").getOrCreate()

df = spark.read.parquet("output/features/engineered_features.parquet")

city_centers = {
    "nyc": (40.7580, -73.9855),
    "london": (51.5074, -0.1278),
    "amsterdam": (52.3676, 4.9041),
    "barcelona": (41.3851, 2.1734),
}

def km_expr(lat_col, lng_col, clat, clng):
    # fast planar approximation in km
    return 111.32 * F.sqrt(
        F.pow(F.col(lat_col) - F.lit(clat), 2) +
        F.pow((F.col(lng_col) - F.lit(clng)) * F.cos(F.radians(F.col(lat_col))), 2)
    )

lu = (df.select("city", "neighbourhood_cleansed", "latitude", "longitude")
        .where(F.col("city").isNotNull() & F.col("neighbourhood_cleansed").isNotNull())
        .groupBy("city", "neighbourhood_cleansed")
        .agg(
            F.avg("latitude").alias("lat"),
            F.avg("longitude").alias("lng"),
            F.count("*").alias("n")
        ))

lu = lu.withColumn("dist_km", F.lit(None).cast("double"))
for c, (clat, clng) in city_centers.items():
    lu = lu.withColumn(
        "dist_km",
        F.when(F.col("city") == c, km_expr("lat", "lng", clat, clng)).otherwise(F.col("dist_km"))
    )

os.makedirs("output/reference", exist_ok=True)
lu.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/reference/neighborhood_lookup")

spark.stop()