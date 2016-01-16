# cdr-topic-gibbs-sampling

Input:
  CDR record data: one text file, each line of which is a single record of call record.
          | UUID of user | time | antenna ID | lat | lon |
          each entry is seperated by '\t'
          
  Antenna index file: one text file, each line of which is a single infomation of an antenna.
          | antenna index | antenna ID | lat | lon |
          each entry is seperated by ','
  
  Antenna distance matrix: the matrix of distance of each pair of antenna i, j (antenna index)
  
Output:
  ntk.csv: the n_{tk} matrix in the cdr topic model. (row is the hour of day [0, 23], column is the k-th topic)
  ndk.csv: the n_{dk} matrix in the cdr topic model. (row is the index of day [0, MAXDAY], column is the k-th topic)
  nulk_k*.csv: the n_{ulk} tensor in the cdr topic model. Each file is the kth slice of the tensor, n_{ul:}. (row is the index of user, column is the index of antenna)
