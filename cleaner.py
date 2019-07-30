#!/usr/bin/env python

import csv 

col_names = ['id', 'date_local', 'caption', 'image_filename', 'tagged_users', 'num_likes', 'comments', 'ad']

with open('post-data.csv') as old_csv_file:
    with open('clean-data.csv', 'w') as new_csv_file:
        of_reader = csv.reader(old_csv_file, delimiter=',')
        nf_writer = csv.writer(new_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        nf_writer.writerow(col_names)
        for row in of_reader:
            nurow = row[:]
            if "#ad" in nurow[2] or "#sponsored" in nurow[2] or "#advertisement" in nurow[2]:
                nurow[2] = nurow[2].replace("#ad", "").replace("#sponsored", "").replace("#advertisement","")
                nurow.extend('1')
            else:
                nurow.extend('0')
            nf_writer.writerow(nurow)
