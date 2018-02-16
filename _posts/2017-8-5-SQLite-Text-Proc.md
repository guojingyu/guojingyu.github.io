---
layout: post
title: Use SQLite in Text-Data Processing
---

In a recent testing development project, I need to generate genetic variant test cases for downstream automated software test. So the test cases will likely be assembled by variants enlisted in [VCF files or Variant Call Format](https://en.wikipedia.org/wiki/Variant_Call_Format) (thinking it as a collection of variant information in a flat file) and their corresponding clinical findings (e.g. disease, genotype, treatment, clinical observation etc). The challenge is for the clinical findings that has underspecified mutations, such as EGFR activating mutations or exon 19 deletion, there is no particular well defined single variant associated that I can use to insert into VCFs. This will have to be done through sampling a whole set of variants with criteria and/or formulate it through gene elements descriptive information.

Well now, good news is that I digged out from [Qiagen (Ingenuity)](https://www.qiagenbioinformatics.com/products/qiagen-clinical-insight/) knowledge base a pretty comprehensive list (as a dump) of known/putative variants with many of them curation validated, bad news is that this list contains >20 million variants, added up to a 2GB flat file with some simple annotation and it is ever-growing. For public, it is OK to use any kind of such variant mapping data repositories such as [dbVar](https://www.ncbi.nlm.nih.gov/dbvar) and [COSMIC](cancer.sanger.ac.uk/). It became rather a problem to solve with script only -- to simply read this file to memory will be a slow process. But what makes this unmanageable is the much larger memory footage and computation cost to build the data structure (hash or dictionary) for this file in memory and to merge it with other file content. The whole process, shall implemented purely by scripting language, while quite possible, can be fragile -- anything happened (all data files to be joined could change) could disrupt the whole process and risk longer running time but to restart. A more persistent solution is required and it’d better be light.

Another requirement to this task is that it will remain as part of the build process for content release from knowledge base and I have to be able to build it as part of a CI process, which is to be triggered by upstream jobs and executable as a script. Since being part of the content build, even on server, this is not supposed to be a memory demanding task or it prevents other jobs from finishing in daily build. 

So I eventually decided to alternatively serve this file from drive along with others. To be able to do so, I utilized SQLite as the persistent layer to serve all files for the manipulation in python. During the run, it turned out to be efficient with up to ~200M memory footage that was quite manageable even on my Macbook Pro and most of the process remained low computation demanding.

### SQLite
A well known name for desktop relational database is SQLite. SQLite runs its databases as files -- meaning the databases can be create, write, and erased as files. But with index properly built, the query performance on single machine is a well match to MySQL. Actually, there is not much need to introduce this tool. SQLite acts a bit like a shell above the database file, or software runs on the file, rather than a server or service.

Install SQLite on Mac with Homebrew is a breeze. 

```bash
$ brew update
$ brew install sqlite3
```
 
In Ubuntu, if not for the latest version of the database, it can be easily installed from Ubuntu repository.

```bash
$ sudo apt-get install sqlite3 libsqlite3-dev
```

To run SQLite (say over a database stored in file mydb.sqlite):

```bash
$ cd /where/the/database/file/is
$ sqlite3 mydb.sqlite
```

This will initialize an interactive shell for one to communicate with the databases, including SQL queries. If the database file did not exist yet, it will be created shall any following update is made on this database (e.g. create table and insert data). Of course, one can start SQLite without indicating the database file but later in the shell, such as,

```shell
$ sqlite3

sqlite> .open mydb.sqlite
```

To quit the shell, do

```shell
sqlite> .exit
```

### SQLite Manager Add-on for Firefox
If to use Firefox browser, there is a very nice little add-on tool named “SQLite Manager” (https://github.com/lazierthanthou/sqlite-manager) can be installed as its add on. It is not necessary for the task but it helps to visualize the databases and test SQL queries. This tool can be installed within Firefox.

![_config.yml]({{ site.baseurl }}/images/2017-8-5-SQLite-Text-Proc/SQLite _Manager.jpg)


### Load file content into Pandas Dataframe
I normally used the Anaconda build for Python and it had pandas included. There is no need to talk much about Pandas either (too much introduction online). Taking the idea from R, pandas is one great dataframe implementation for table-like data processing. 

Loading data to a pandas dataframe from a flat file is a no brainer (just like R):

```python
import pandas as pd
df = pd.read_table(file, header=0)  # assuming header exists and tab delimited
```

But to load the big file, the best practice is to avoid loading it at once or any intermediate operation (e.g. column trimming, splitting etc.) during the loading can lock up the process -- it may not be dead but too slow without response. The robust way to do it is through DataFrame chunk:

```python
chunksize = 10 ** 5 ## indicate 100K row per chunk for 200M memory use or do what you'd like
chunk_counter = 0
for chunk in pd.read_table(file, header=0, chunksize=chunksize):
    ## ************
    ## process the dataframe chunk as you need
    ## or loading to database
    ## ************
    chuck_counter += 1
```

Each chunk can be just treated as a “mini” dataframe of the original file. So most Pandas dataframe functions or series functions can work directly on chunk object or its column trouble free.

Just a side note, for text processing, Pandas Dataframe is very handy. But for numeric computing, Pandas Dataframe may not be a good choice. What fits better for matrix-like numeric computing is numpy ndarray -- personal experience suggested up to 100 times faster in these applications.

### Loading to SQLite
Another reason to choose Pandas Dataframe is that it has build-in connection operators for database. While plain python can do it easily, such as

```python
import sqlite3 as sqlite

class SQLRunner(object):
   @staticmethod
   def runSQL(db, query):
       """
       Accept SQL query and return query result
       :param query:
       :return:
       """
       result = None
       try:
           conn = sqlite.connect(db)
           with conn:
               cur = conn.cursor()
               result = cur.execute(query)
               conn.commit()

       except sqlite.Error, e:
           print "Error: %s" % e.args[0]
           print "None object returned."

       finally:
           if conn:
               conn.close()
       return result
```

It cannot match the pythonic simplicity of the mighty Pandas Dataframe (df):

```python
conn = sqlite.connect(db)
df.to_sql(table, conn, flavor='sqlite', if_exists = 'replace', index = index)
```
Or 

```python
df.to_sql(table, conn, flavor='sqlite', if_exists = 'append', index = index)
```

I tried to load freshly for each table so I will “replace” existing table by the first chunk and “append” by the following chunks.

### Query Planning (Optimization)
Before any proper query planning, disappointingly, it took ~10s for the database to return the result for one single variant lookup, which is comprised of 3 separate SQL queries. But the point for using a database is we can make plan for how to query the tables so to speed up. One way to do it is to indexing tables -- just like to create an index for a book or dictionary. 

Relational database index is commonly indexed as a derivative tree structure (e.g. B-tree). This would reduce linear search by row or column to binary search by index (logarithmic time). The better part is to make multiple column index so that queries over multiple columns can be done by searching single index. To create index, insert a query like this after a table is loaded before the connector is closed.

```python
connector.SQLRunner.runSQL(sqlite_db, "CREATE INDEX did_refseq_id_idx ON %s (id, refseq_id)" % table_name)
```

Then query it on the table still:

```python
cur.execute("SELECT gene_symbol, refseq_id FROM %s WHERE id = '%s' AND refseq_id > 'NO' AND refseq_id < 'NQ'" % (table_name, id))
```

Create/query the index can be a bit tricky here:

1. The logical expression order for each column in WHERE clause of select SQL has to be in the same order as in the index creating SQL query (i.e. “id”, “refseq\_id” but not reversed); 

2. To specify a string as Refseq protein id as “NP\_00000.0” (to say starting by “NP”), one cannot use the pattern query clause term “LIKE” such as “refseq\_id LIKE ‘NP%’”. It has to be converted to specific criteria such as inequalities of some kind, such as refseq\_id > 'NO' AND refseq\_id < 'NQ' (meaning not before “NO” and not after “NQ”, exclusively, so it is “NP*”). Please see [Refseq](https://en.wikipedia.org/wiki/RefSeq) for more detailed information, but in general the given example is just a way to form a range query that can utilize the index (see [here](http://use-the-index-luke.com/sql/where-clause/searching-for-ranges/greater-less-between-tuning-sql-access-filter-predicates) for more information).

After indexed the tables and optimized the SQL queries, the same single variant lookup with three SQL queries can be returned in <0.005 s on my 2-year-old laptop. So it takes ~10 seconds to finish more than 1000 variant lookup sessions, which would otherwise use more than 3 hours. 

### Wrap up the project and retro
At last, while dockerizing this project is not an explicit point by this article, docker helped to simplified the deployment, since the whole job can be run on a single docker container with python (pandas lib etc) and SQLite database pre-installed in the image. The to-be-imported datafile will be first exported from knowledge base or other sources to a folder that is mounted and readable to the virtual machine/docker container. 

The overall process can be done by ~10 minutes for data parsing and uploading ~2.5GB data files, and ~2 minutes for queries, with ~10K test cases (writing to VCFs etc) -- well, besides whatever upstream knowledge base query time and the docker initialization.

#### Some retro here:
1. I did not normalize the tables within the database -- for such a project with several tables involved, such normalization may not help as much on the query performance. Also because hard drive is much cheaper to use, so serving a project that is very single purposed, normalization over 10 tables to prevent redundancy does not seem to be a time-wise worthy move. 

2. There can be some QC work and constraints to add to make the process more robust and reduce the data error -- but this can be incrementally added when the whole setup is stabilized.

3. SQLite seems to be quite handy and lightweight. I like its fitness for research/preliminary level projects, where no absolute performance but flexibility is needed. This project can be run at full just on my laptop computer and yet even with a 20GB file, it can still be manageable (longer loading overhead but only slightly slower queries given logarithmic complexity).

4. Database table join could be another optimization. I did not join the tables mainly because there are more logical processing involved that is harder for SQL but easy for python. It is also because the query number per run is bounded to 10K or so, but there is the single 20M row table that slows down every join while most of the joined rows will not be visited in that run. So not an efficient choice.

5. If implementing in python dictionary structure then lookup, this project will become much more computation demanding (not mentioning more complex and less maintainable code logic). While for this implementation, the computation can be shortly heavy while building the indices for tables (for some seconds) but most other occasion is very light.
