# Frequent Itemsets

## Apriori
### Items are assumed to be sorted, and positive integers

os.system(f"java -jar spmf.jar run Apriori ./inputs/corrected.in ./outputs/out.out {minsup}")

---
## AprioriTID_Bitset 
### show_transaction_ids 
### Parameters: show transaction ids: True/False | Max Pattern Length: Integer
#### uses bitsets as internal structures instead of HashSet of Integers to represent sets of transactions IDs. The advantage of the bitset version is that using bitsets for representing sets of transactions IDs is more memory efficient and performing the intersection of two sets of transactions IDs is more efficient with bitsets

os.system(f"java -jar spmf.jar run AprioriTID_Bitset ./inputs/in.in ./outputs/out.out {minsup} True")
---
## FP-Growth
### Parameters: Max pattern length (integer) [infinity] | Minimum pattern length (integer) [1]
### transactions are assumed to be sorted, items positive integers

os.system(f"java -jar spmf.jar run FPGrowth_itemsets ./inputs/in.in ./outputs/out.out {minsup} 3 2")   # first max then min

---
## Eclat/dEclat
### uses depth-first search
### Eclat Parameters : Show transaction ids (true/false) | Max pattern length (integer) [infinity]
### transaction DB should be sorted
### items are positive integers
### dEclat don't have transaction id

os.system(f"java -jar spmf.jar run Eclat ./inputs/in.in ./outputs/out.out {minsup} True 3") 

#os.system(f"java -jar spmf.jar run dEclat ./inputs/in.in ./outputs/out.out {minsup} 3")   # dEclat don't have transaction id

---
## H-Mine: H-Mine uses a pattern-growth approach
#### parameters: Max pattern length (integer) [infinity]


os.system(f"java -jar spmf.jar run HMine ./inputs/corrected.in ./outputs/out.out {minsup} 3")
---
## FIN 
#### very recent algorithm (2014) It is very fast.

os.system(f"java -jar spmf.jar run FIN ./inputs/corrected.in ./outputs/out.out {minsup}")  
---
## DFIN 
#### improved version of FIN

os.system(f"java -jar spmf.jar run DFIN ./inputs/corrected.in ./outputs/out.out {minsup}")  
---
## NegFIN
#### NegFIN is a very recent algorithm (2018)
#### very fast on MSCRED

os.system(f"java -jar spmf.jar run NegFIN ./inputs/corrected.in ./outputs/out.out {minsup}") 
---
## PrePost / PrePost+
#### 2012-2015 

#os.system("java -jar spmf.jar run PrePost ./inputs/in.in ./outputs/out.out 60%") 

os.system(f"java -jar spmf.jar run PrePost+ ./inputs/corrected.in ./outputs/out.out {minsup}") 
---
## LCMFreq
### It is supposed to be one of the fastest itemset mining algorithm.
### attempted to replicate LCM v2 used in FIMI 2004
### Parameters: Max pattern length (integer) [infinity]
### very fast on MSCRED, needs sorting otherwise wrong results


os.system(f"java -jar spmf.jar run LCMFreq ./inputs/corrected.in ./outputs/out.out {minsup} 3") 
---
# Frequent Closed Itemsets
- A frequent closed itemset is a frequent itemset that is not included in a proper superset having exactly the same support. The set of frequent closed itemsets is thus a subset of the set of frequent itemsets.
- No information is lost
---
## AprioriClose
### (historically) the first algorithm for mining frequent closed itemsets.
### QUESTION: whats wrong with toy outputs?

os.system(f"java -jar spmf.jar run AprioriClose ./inputs/in.in ./outputs/out.out {minsup}") 
---
## DCI_Closed
### one of the fastest
### Parameters: "show transaction ids" (true/false) 
### Cannot handle 0 
### support is integer (support count)
### QUESTION: why there is wrong TID (e.g., for 3D item in toy)

  
os.system(f"java -jar spmf.jar run DCI_Closed ./inputs/corrected.in ./outputs/out.out {support_count} true")  
---
## Charm
### Charm is an important algorithm because it is one of the first depth-first algorithm for frequent closed itemsets. In SPMF, Charm and DCI_Closed are the two most efficient algorithms for frequent closed itemset mining.
### dCharm is a variation of the Charm that is implemented with diffsets rather than tidsets.
### Parameters: "show transaction ids" (true/false) 
### CORRECT ANSWER ON TOY

os.system(f"java -jar spmf.jar run Charm_bitset ./inputs/in.in ./outputs/out.out {minsup} True")
---
## LCM
### frequent closed itemsets version
### attempted to replicate LCM v2 used in FIMI 2004. Most of the key features of LCM have been replicated
### needs sorted transactions

os.system(f"java -jar spmf.jar run LCM ./inputs/corrected.in ./outputs/out.out {minsup}")
---
## FPClose
### FPClose is one of the fastest in the FIMI 2004 competition
### did not implement the triangular matrix from FPGrowth* and the local CFI trees. These optimizations may be added in a future version of SPMF.

os.system(f"java -jar spmf.jar run FPClose ./inputs/in.in ./outputs/out.out {minsup}")
---
## NAFCP (2015)
#### It internally uses a structure called N-List.

os.system(f"java -jar spmf.jar run NAFCP ./inputs/in.in ./outputs/out.out {minsup}")
---
## NEclatClosed (2021)
#### uses N-lists
#### Newest

os.system(f"java -jar spmf.jar run NEclatClosed ./inputs/in.in ./outputs/out.out {minsup}")
---
# perfectly rare itemsets
#### The output of AprioriInverse is the set of all perfectly rare itemsets in the database such that their support is lower than maxsup and higher than minsup.
#### AprioriInverse is the only algorithm for perfectly rare itemset mining offered in SPMF. Since it is based on Apriori, it suffers from the same fundamental limitations
#### There is an alternative implementation of AprioriInverse in SPMF called "AprioriInverse_TID". It shows transaction ids.

# seems not vary robust, needs more testing if use is justified

os.system(f"java -jar spmf.jar run AprioriInverse ./inputs/in.in ./outputs/out.out {minsup} {maxsup}")

os.system(f"java -jar spmf.jar run AprioriInverse_TID ./inputs/in.in ./outputs/out.out {minsup} {maxsup} True")


---
# Frequent Maximal Itemsets
- A frequent maximal itemset is a frequent itemset that is not included in a proper superset that is a frequent itemset. The set of frequent maximal itemsets is thus a subset of the set of frequent closed itemsets, which is a subset of frequent itemsets.
- frequent maximal itemsets are not a lossless representation of the set of frequent itemsets (it is possible to regenerate all frequent itemsets from the set of frequent maximal itemsets but it would not be possible to get their support without scanning the database).
---
## FPMax
#### Based on FPGrowth
#### The FPMax algorithm is a very efficient algorithm for maximal itemset mining.
#### This seems the answer to our problem!

os.system(f"java -jar spmf.jar run FPMax ./inputs/in.in ./outputs/out.out {minsup}")
---
# Charm_MFI (2006)
# Optional Parameters: "show transaction ids" (true/false) 
# Charm-MFI is not an efficient algorithm because it discovers maximal itemsets by performing post-processing after discovering frequent closed itemsets with the Charm algorithm (hence the name: Charm-MFI).

os.system(f"java -jar spmf.jar run Charm_MFI ./inputs/in.in ./outputs/out.out {minsup} True")
---
# Generator Itemsets
- A generator is an itemset X such that there does not exist an itemset Y strictly included in X that has the same support.
- Are generators useful for this problem?
---
## DefMe
#### Optional parameters: Max pattern length [infinity]
#### DefMe is an algorithm proposed at PAKDD 2014 for discovering minimal patterns in set systems. If it is applied to itemset mining, it will discover frequent itemset generator. In SPMF, we have implemented it for this purpose.

os.system(f"java -jar spmf.jar run DefMe ./inputs/in.in ./outputs/out.out {minsup} 4")
---
## Pascal
#### Optional Paremeter: Max pattern length (integer)
#### Pascal is an Apriori-based algorithm. 
# Discovering frequent itemsets and at the same time identify which ones are generators in a transaction database.

os.system(f"java -jar spmf.jar run Pascal ./inputs/in.in ./outputs/out.out {minsup} 3")
---
## Zart
#### Zart is an algorithm for discovering frequent closed itemsets and their corresponding generators in a transaction database.
#### Zart is an Apriori-based algorithm. Why is it useful to discover closed itemsets and their generators at the same time? One reason is that this information is necessary to generate some special kind of association rules such as the IGB basis of association rules

os.system(f"java -jar spmf.jar run Zart ./inputs/in.in ./outputs/out.out {minsup}")
---
# Minimal Rare Itemsets
- A minimal rare itemset is an itemset that is not a frequent itemset and that all its subsets are frequent itemsets.
- Are generators useful for this problem?
---
## AprioriRare
#### Optional parameters: "show transaction ids?" (true/false)
#### AprioriRare is the only algorithm for minimal rare itemset mining offered in SPMF. 

os.system(f"java -jar spmf.jar run AprioriRare ./inputs/in.in ./outputs/out.out {minsup}")
---
# High Utility Itemsets
- fill empty transactions like this:
  - 1000:0:0
- they are generally slower

---
## Two-Phase
#### Optional parameters: minimum utility threshold min_utility (a positive integer)
#### It is assumed that all items within a same transaction (line) are sorted according to a total order (e.g. ascending order) and that no item can appear twice within the same transaction.
#### application:  discovering groups of items in transactions of a store that generate the most profit.

os.system(f"java -jar spmf.jar run Two-Phase ./inputs/utility.in ./outputs/out.out {minutil}")
---
## FHM
#### It is assumed that all items within a same transaction (line) are sorted according to a total order
#### The FHM algorithm was shown to be up to six times faster than HUI-Miner (also included in SPMF), especially for sparse datasets (see the performance section of the website for a comparison). But the EFIM algorithm (also included in SPMF) greatly outperforms FHM (see performance section of the website).

os.system(f"java -jar spmf.jar run FHM ./inputs/utility.in ./outputs/out.out {minutil}")

---
## EFIM
#### one of the most efficient
#### The EFIM algorithm was shown to be up to two orders of magnitude faster than the previous state-of-the-art algorithm FHM, HUI-Miner, d2HUP, UPGrowth+ (also included in SPMF), and consumes up to four times less memory (see the performance section of the website for a comparison).

os.system(f"java -jar spmf.jar run EFIM ./inputs/utility.in ./outputs/out.out {minutil}")
---
## CHUI-Miner(Max)
#### Maximal high utility itemsets
#### CHUI-Miner(Max) (Wu et al., 2019) is an algorithm for discovering maximal high-utility itemsets in a transaction database containing utility information.

os.system(f"java -jar spmf.jar run CHUI-MinerMax ./inputs/utility.in ./outputs/out.out {minutil}")

