def create_spark():
    spark = SparkSession.builder \
        .appName("AccidentsLogisticRegression") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print("📥 Đã load dữ liệu:", df.count(), "dòng")
    return df