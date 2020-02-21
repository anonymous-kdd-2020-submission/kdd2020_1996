dataset=NYT

# text file name; one document per line
text_file=corpus_token.txt

taxo_file=mat_taxo_mini.txt

neg_file=subtree_taxo_mini.txt

level_file=level_taxo_mini.txt

# tree node embedding output file name
tree_emb_file=tree_emb.txt

# word embedding output file name
out_file=emb.txt

# word embedding dimension
word_dim=100

# local context window size
window_size=5

# minimum word count in corpus; words that appear less than this threshold will be discarded
min_count=5

# number of iterations to run on the corpus
iter=5

# number of threads to be run in parallel
threads=10

cd ./src
make sphere_tree
cd ..


./src/sphere_tree -train ${dataset}/${text_file} -level-file ${dataset}/${level_file} -taxo-file ${dataset}/${taxo_file} \
	-output ${dataset}/${out_file} -tree-emb ${dataset}/${tree_emb_file} -neg-file ${dataset}/${neg_file} \
	-size ${word_dim} -window ${window_size} -sample 1e-3 -word-margin 0.25 -tree-margin 0.35 -tree-margin-sub 0.18 -cat-margin 0.9 \
	-alpha 0.025 -tree-period 256 -global-lambda 1.5 -lambda-cat 1.0 -lambda-tree 1.0 -negative 2 \
	-expand 1 -pre-iter 0 \
	-min-count ${min_count} -iter ${iter} -threads ${threads}