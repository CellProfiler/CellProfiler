import MySQLdb
import re

# Initialize variables
database_name = "2009_08_25_JamesBradner_Autophagy"
per_image_table_name = "Per_Image"
per_object_table_name = "Per_Object"
columns_to_drop = "Mean_|Median_"

# Establish connection
db = MySQLdb.connect(host='imgdb01', user='cpuser', passwd='cPus3r', db='%s'%(database_name))
cursor = db.cursor()

# Get column names
query = "select column_name from information_schema.columns where table_name = '%s' and table_schema = '%s';"%(per_image_table_name,database_name)
cursor.execute(query)
column_names = cursor.fetchall()

# Construct query statement for column drop
query = ""
for c in column_names:
	if re.match('(%s)'%(columns_to_drop),c[0]):
		query = query + 'alter table %s drop column %s; '%(per_image_table_name, c[0])

cursor.execute(query)
print "Done"