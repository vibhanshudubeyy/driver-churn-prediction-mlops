from sqlalchemy import create_engine, text

engine = create_engine("mysql+mysqlconnector://root:root@localhost:3306/driver_churn_db")

conn = engine.connect()

result = conn.execute(text("SELECT DATABASE()"))

for row in result:
    print(row)