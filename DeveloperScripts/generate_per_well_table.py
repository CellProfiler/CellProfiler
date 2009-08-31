import MySQLdb
import re

# Initialize variables
database_name = "2009_08_25_JamesBradner_Autophagy"
per_image_table_name = "Per_Image"
per_object_table_name = "Per_Object"
metadata_plate = "Image_Metadata_Run"
metadata_well = "Image_Metadata_Well"
table_name_prefix = re.split('Per_Image',per_image_table_name)[0]

# Establish connection
db = MySQLdb.connect(host='imgdb01', user='cpuser', passwd='cPus3r', db='%s'%(database_name))
cursor = db.cursor()

# Create a temporary table
query = "drop table if exists %sTemp; "%(table_name_prefix)
cursor.execute(query)
query = "create table %sTemp as select * from %s left join %s using (ImageNumber);"%(table_name_prefix, per_image_table_name, per_object_table_name)
print "Creating temporary table...",
cursor.execute(query)
print "Done"

# Get column names
query = "select column_name from information_schema.columns where table_name = '%sTemp' and table_schema = '%s';"%(table_name_prefix,database_name)
cursor.execute(query)
column_names = cursor.fetchall()

# Create new table with proper column names
query = "drop table if exists %sPer_Well"%(table_name_prefix)
cursor.execute(query)
query = "create table if not exists %sPer_Well like %sTemp;"%(table_name_prefix,table_name_prefix)
cursor.execute(query)

# Construct query statement for aggregation
query = "insert into %sPer_Well select "%(table_name_prefix)

for c in column_names:
	if re.search('(TableNumber|ImageNumber|PathName|FileName|Metadata)',c[0]):
		query = query + '%s, '%(c[0])
	elif re.search('Count',c[0]):
		query = query + 'sum(%s), '%(c[0])
	else:
		query = query + 'avg(%s), '%(c[0])

# Get rid of trailing comma and space from query
query = query[0:-2]

# Tack on additional grouping syntax and execute
query = query + ' from %sTemp group by %s, %s order by %s, %s;'%(table_name_prefix,metadata_plate,metadata_well,metadata_plate,metadata_well)
print "Creating per-well table...",
cursor.execute(query)
print "Done"

# Drop the temp table
query = "drop table %sTemp;"%(table_name_prefix)
print "Dropping temporary table...",
cursor.execute(query)
print "Done"