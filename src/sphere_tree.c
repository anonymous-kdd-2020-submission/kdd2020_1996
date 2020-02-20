//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_STRING 100
#define MAX_WORDS_NODE 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int corpus_max_size = 40000000;  // Maximum 40M documents in the corpus

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct tree_node {
  // emb: node embedding
  real *emb, *wt_score;
  // *dist: array of tree distance to other nodes
  // *cur_words: array of vocabulary indices of current retrieved representative words
  // init_size: how many seed words are given
  // cur_size: total number of current retrieved representative words
  int *dist, node_id, *cur_words, init_size, cur_size, *neg_ids, level;
  char *node_name;
};

char train_file[MAX_STRING], output_file[MAX_STRING], tree_emb_file[MAX_STRING], neg_file[MAX_STRING], level_file[MAX_STRING];
char save_vocab_file[MAX_STRING], load_emb_file[MAX_STRING], read_vocab_file[MAX_STRING], taxo_file[MAX_STRING];
struct vocab_word *vocab;
struct tree_node *tree;
long long *doc_sizes;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int num_per_topic = 10; // top-k words per topic to show
int *vocab_hash, *docs;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, pretrain_iter = 0, file_size = 0, classes = 0, iter_count;
real alpha = 0.025, starting_alpha, sample = 1e-3, global_lambda = 1.5, lambda_tree = 1.0, lambda_cat = 1.0;
real word_margin = 0.15, tree_margin = 0.2, tree_margin_sub = 0.2, cat_margin = 0.2;
real *syn0, *syn1, *syn1neg, *syn1doc, *expTable, *wt_score_ptr;
clock_t start;
int *rankings;

// how many words to pass before embedding tree
int tree_emb_period = 128;
// number of nodes in the tree
long long nodes;

int hs = 0, negative = 5, expand = 1;
const int table_size = 1e8;
int *table, *doc_table;


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void InitDocTable() {
  int a, i;
  double doc_len_pow = 0;
  double d1, power = 0.75;
  doc_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < corpus_size; a++) doc_len_pow += pow(docs[a], power);
  i = 0;
  d1 = pow(docs[i], power) / doc_len_pow;
  for (a = 0; a < table_size; a++) {
    doc_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(docs[i], power) / doc_len_pow;
    }
    if (i >= corpus_size) i = corpus_size - 1;
  }
}

int IntCompare(const void * a, const void * b) { 
  return *(int*)a - *(int*)b; 
}

int SimCompare(const void *a, const void *b) { // large -> small
  return (wt_score_ptr[*(int *) a] < wt_score_ptr[*(int *) b]) - (wt_score_ptr[*(int *) a] > wt_score_ptr[*(int *) b]);
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocab(word);
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Record document length
void ReadDoc(FILE *fin) {
  char word[MAX_STRING], eof = 0;
  long long i;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    i = SearchVocab(word);
    if (i == 0) {
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
    } else if (i == -1) continue;
    else docs[corpus_size]++;
  }
  // for (i = 0; i <= 5; i++) printf("%lld\n", doc_sizes[i]);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING], eof = 0;
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    train_words++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    }
    else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fseek(fin, 0, SEEK_SET);
  ReadDoc(fin);
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void ReadTaxo() {
  long long a, i, j, no_neg = 0;
  unsigned long long next_random = 1;
  real norm = 0.0;

  // Read taxo file
  FILE *f = fopen(taxo_file, "rb");
  printf("Negative node file: %s\n", taxo_file);
  if (f == NULL) {
    printf("Taxonomy file not found\n");
    exit(1);
  }

  // Read negative indices
  FILE *f_neg = fopen(neg_file, "rb");
  printf("Negative node file: %s\n", neg_file);
  if (f_neg == NULL) {
    printf("Negative node file not found! Assume all nodes are negative!\n");
    exit(1);
  }

  // Read node level
  FILE *f_level = fopen(level_file, "rb");
  printf("Node level file: %s\n", level_file);
  if (f_neg == NULL) {
    printf("Node level file not found! Assume all nodes are negative!\n");
    exit(1);
  }
  fscanf(f, "%lld", &nodes);
  printf("%lld\n", nodes);
  if (!no_neg) fscanf(f_neg, "%lld", &nodes);
  fscanf(f_level, "%lld", &nodes);
  tree = (struct tree_node *)calloc(nodes, sizeof(struct tree_node));
  rankings = (int *)calloc(vocab_size, sizeof(real));
  for (a = 0; a < vocab_size; a++) rankings[a] = a;
  for (i = 0; i < nodes; i++) {
    tree[i].node_name = (char *)calloc(MAX_STRING, sizeof(char));
    tree[i].cur_words = (int *)calloc(MAX_WORDS_NODE, sizeof(int));
    tree[i].dist = (int *)calloc(nodes, sizeof(int));
    tree[i].emb = (real *)calloc(layer1_size, sizeof(real));
    tree[i].wt_score = (real *)calloc(vocab_size, sizeof(real));
    tree[i].neg_ids = (int *)calloc(nodes, sizeof(int));
    
    fscanf(f, "%s", tree[i].node_name);
    printf("node name: %s\n", tree[i].node_name);

    // Skip ROOT node
    if (i > 0) {
      a = SearchVocab(tree[i].node_name);
      if (a == -1) {
        printf("Category name %s not in vocabulary!\n", tree[i].node_name);
        exit(1);
      }
      tree[i].cur_words[0] = a;
      tree[i].init_size = 1;
      tree[i].cur_size = 1;
      for (j = 0; j < tree[i].cur_size; j++) {
        int word = tree[i].cur_words[j];
        for (a = 0; a < layer1_size; a++) tree[i].emb[a] += syn0[a + word*layer1_size];
      }
      norm = 0.0;
      for (a = 0; a < layer1_size; a++) 
        norm += tree[i].emb[a] * tree[i].emb[a];
      for (a = 0; a < layer1_size; a++)
        tree[i].emb[a] /= sqrt(norm);
    }
    else {
      tree[i].cur_size = 0;
      norm = 0.0;
      for (a = 0; a < layer1_size; a++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        tree[i].emb[a] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += tree[i].emb[a] * tree[i].emb[a];
      }
      for (a = 0; a < layer1_size; a++)
        tree[i].emb[a] /= sqrt(norm);
    }

    tree[i].node_id = i;
    for (j = 0; j < nodes; j++) {
      fscanf(f, "%d", &tree[i].dist[j]);
      // printf("%d ", tree[i].dist[j]);
    }
    // printf("\n");
    if (!no_neg) {
      for (j = 0; j < nodes; j++) {
        fscanf(f_neg, "%d", &tree[i].neg_ids[j]);
        // printf("%d ", tree[i].neg_ids[j]);
      }
    }
    else {
      for (j = 0; j < nodes; j++) tree[i].neg_ids[j] = 1;
    }
    fscanf(f_level, "%d", &tree[i].level);
    // printf("%d ", tree[i].level);
    // printf("\n");
  }
  fclose(f);
  if (!no_neg) fclose(f_neg);
  fclose(f_level);
}

void LoadEmb(char *emb_file, real *emb_ptr) {
  long long a, b;
  int *vocab_match_tmp = (int *) calloc(vocab_size, sizeof(int));
  int vocab_size_tmp = 0, word_dim;
  char *current_word = (char *) calloc(MAX_STRING, sizeof(char));
  real *syn_tmp = NULL, norm;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn_tmp, 128, (long long) layer1_size * sizeof(real));
  if (syn_tmp == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  printf("Loading embedding from file %s\n", emb_file);
  if (access(emb_file, R_OK) == -1) {
    printf("File %s does not exist\n", emb_file);
    exit(1);
  }
  // read embedding file
  FILE *fp = fopen(emb_file, "r");
  fscanf(fp, "%d", &vocab_size_tmp);
  fscanf(fp, "%d", &word_dim);
  if (layer1_size != word_dim) {
    printf("Embedding dimension incompatible with pretrained file!\n");
    exit(1);
  }
  vocab_size_tmp = 0;
  while (1) {
    fscanf(fp, "%s", current_word);
    a = SearchVocab(current_word);
    if (a == -1) {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &syn_tmp[b]);
    }
    else {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &emb_ptr[a * layer1_size + b]);
      vocab_match_tmp[vocab_size_tmp] = a;
      vocab_size_tmp++;
    }
    if (feof(fp)) break;
  }
  printf("In vocab: %d\n", vocab_size_tmp);
  qsort(&vocab_match_tmp[0], vocab_size_tmp, sizeof(int), IntCompare);
  vocab_match_tmp[vocab_size_tmp] = vocab_size;
  int i = 0;
  for (a = 0; a < vocab_size; a++) {
    if (a < vocab_match_tmp[i]) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        emb_ptr[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += emb_ptr[a * layer1_size + b] * emb_ptr[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        emb_ptr[a * layer1_size + b] /= sqrt(norm);
    }
    else if (i < vocab_size_tmp) {
      i++;
    }
  }
  fclose(fp);
  free(current_word);
  free(emb_file);
  free(vocab_match_tmp);
  free(syn_tmp);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  real norm;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
  a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
  a = posix_memalign((void **) &syn1doc, 128, (long long) corpus_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed (syn0)\n");
    exit(1);
  }
  if (syn1neg == NULL) {
    printf("Memory allocation failed (syn1neg)\n");
    exit(1);
  }
  if (syn1doc == NULL) {
    printf("Memory allocation failed (syn1doc)\n");
    exit(1);
  }
  
  if (load_emb_file[0] != 0) {
    char *center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    strcpy(center_emb_file, load_emb_file);
    LoadEmb(center_emb_file, syn0);
  }
  else {
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn1neg[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += syn1neg[a * layer1_size + b] * syn1neg[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] /= sqrt(norm);
    }
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += syn0[a * layer1_size + b] * syn0[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn0[a * layer1_size + b] /= sqrt(norm);
    }
  }
  for (a = 0; a < corpus_size; a++) {
    norm = 0.0;
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      syn1doc[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      norm += syn1doc[a * layer1_size + b] * syn1doc[a * layer1_size + b];
    }
    for (b = 0; b < layer1_size; b++)
      syn1doc[a * layer1_size + b] /= sqrt(norm);
  }

  ReadTaxo();
}

// compute tree embedding loss with current node embedding and perform gradient descent to update node embeddings
real TreeEmb() {
  int u, v, vp, a, c, cnt = 0;
  real f, g, h, loss = 0, cur_margin = 0;
  int *negs = (int*) malloc(nodes * sizeof(int));
  int parent = 0;
  real *grad = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  // for (u = 0; u < nodes; u++) {
  //   if (tree[u].level == 1) cur_margin = tree_margin;
  //   else if (tree[u].level == 2) cur_margin = tree_margin_sub;
  //   for (v = 0; v < nodes; v++) {
  //     if (u == v) continue;
  //     int tree_dist = tree[u].dist[v];
  //     // find all other nodes with dist > tree_dist
  //     int neg_count = 0;
  //     for (vp = 0; vp < nodes; vp++) {
  //       if (vp == u || vp == v) continue;
  //       if (tree[u].dist[vp] > tree_dist) negs[neg_count++] = vp;
  //     }
  //     // if (neg_count < 5){
  //     //   continue;
  //     // }
  //     cnt ++;
  //     // compute loss with embeddings

  //     f = 0;
  //     for (c = 0; c < layer1_size; c++) f += tree[u].emb[c] * tree[v].emb[c];
  //     for (a = 0; a < neg_count; a++) {
  //       vp = negs[a];
  //       h = 0;
  //       for (c = 0; c < layer1_size; c++) h += tree[u].emb[c] * tree[vp].emb[c];

  //       if (f - h < cur_margin) {
  //         loss += cur_margin - (f - h);
  //         // compute center tree node gradient
  //         for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
  //         for (c = 0; c < layer1_size; c++) neu1e[c] += tree[v].emb[c] - f * tree[u].emb[c] 
  //                                                     + h * tree[u].emb[c] - tree[vp].emb[c];
          
  //         // update positive tree node
  //         for (c = 0; c < layer1_size; c++) grad[c] = tree[u].emb[c] - f * tree[v].emb[c];
  //         for (c = 0; c < layer1_size; c++) tree[v].emb[c] += alpha * lambda_tree / nodes * grad[c];
  //         g = 0;
  //         for (c = 0; c < layer1_size; c++) g += tree[v].emb[c] * tree[v].emb[c];
  //         for (c = 0; c < layer1_size; c++) tree[v].emb[c] /= sqrt(g);

  //         // update negative tree node
  //         for (c = 0; c < layer1_size; c++) grad[c] = h * tree[vp].emb[c] - tree[u].emb[c];
  //         for (c = 0; c < layer1_size; c++) tree[vp].emb[c] += alpha * lambda_tree / nodes * grad[c];
  //         g = 0;
  //         for (c = 0; c < layer1_size; c++) g += tree[vp].emb[c] * tree[vp].emb[c];
  //         for (c = 0; c < layer1_size; c++) tree[vp].emb[c] /= sqrt(g);

  //         // update center tree node
  //         for (c = 0; c < layer1_size; c++) tree[u].emb[c] += alpha * lambda_tree / nodes * neu1e[c];
  //         g = 0;
  //         for (c = 0; c < layer1_size; c++) g += tree[u].emb[c] * tree[u].emb[c];
  //         for (c = 0; c < layer1_size; c++) tree[u].emb[c] /= sqrt(g);
  //       }
  //     }
  //   }
  // }


  for (u = 1; u < nodes; u++) {
    int neg_count = 0;
    for (v = 0; v < nodes; v++) {
      if (tree[u].neg_ids[v] == 2)
        parent = v;
      else if (tree[u].neg_ids[v] == 1 && u != v)
        negs[neg_count++] = v;
    }
    f = 0;
    // printf("parent: %d\n", parent);
    // real cur_margin = tree_margin - (tree[u].level - 1) * 0.1;
    if (tree[u].level == 1) cur_margin = tree_margin;
    else if (tree[u].level == 2) cur_margin = tree_margin_sub;
    for (c = 0; c < layer1_size; c++) f += tree[u].emb[c] * tree[parent].emb[c];
    for (a = 0; a < neg_count; a++) {
      v = negs[a];
      // printf("neg: %d\n", v);
      h = 0;
      for (c = 0; c < layer1_size; c++) h += tree[u].emb[c] * tree[v].emb[c];
      // printf("%f\n", tree_margin * 1 / tree[u].level);

      if (f - h < cur_margin) {
        loss += cur_margin - (f - h);
        // compute center tree node gradient
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] += tree[parent].emb[c] - f * tree[u].emb[c] 
                                                    + h * tree[u].emb[c] - tree[v].emb[c];
        
        // update positive tree node
        for (c = 0; c < layer1_size; c++) grad[c] = tree[u].emb[c] - f * tree[parent].emb[c];
        for (c = 0; c < layer1_size; c++) tree[parent].emb[c] += alpha * lambda_tree / nodes * grad[c];
        g = 0;
        for (c = 0; c < layer1_size; c++) g += tree[parent].emb[c] * tree[parent].emb[c];
        for (c = 0; c < layer1_size; c++) tree[parent].emb[c] /= sqrt(g);

        // update negative tree node
        for (c = 0; c < layer1_size; c++) grad[c] = h * tree[v].emb[c] - tree[u].emb[c];
        for (c = 0; c < layer1_size; c++) tree[v].emb[c] += alpha * lambda_tree / nodes * grad[c];
        g = 0;
        for (c = 0; c < layer1_size; c++) g += tree[v].emb[c] * tree[v].emb[c];
        for (c = 0; c < layer1_size; c++) tree[v].emb[c] /= sqrt(g);

        // update center tree node
        for (c = 0; c < layer1_size; c++) tree[u].emb[c] += alpha * lambda_tree / nodes * neu1e[c];
        g = 0;
        for (c = 0; c < layer1_size; c++) g += tree[u].emb[c] * tree[u].emb[c];
        for (c = 0; c < layer1_size; c++) tree[u].emb[c] /= sqrt(g);
      }
    }
  }
  free(negs);
  free(grad);
  free(neu1e);
  return loss / nodes;
}

// embed representative words close to corresponding category node
real CatEmb() {
  long long i, j, k, word, c, cnt = 0;
  real f, g, h, loss = 0;
  real *grad = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  // real sum_exp;
  // real *exp_list = (real *)calloc(nodes, sizeof(real));
  // for (i = 1; i < nodes; i++)
  //   for (j = 0; j < tree[i].cur_size; j++) {
  //     word = tree[i].cur_words[j];
  //     sum_exp = 0;
  //     for (k = 0; k < nodes; k++) {
  //       if (tree[i].neg_ids[k] > 0 || i == k) {
  //         f = 0;
  //         for (c = 0; c < layer1_size; c++) f += syn0[c + word*layer1_size] * tree[k].emb[c];
  //         exp_list[k] = exp(f);
  //         sum_exp += exp(f);
  //       }
  //     }
  //     loss += -log(exp_list[i]) + log(sum_exp);
  //     // printf("num: %f dem: %f loss: %f\t", exp_list[i], sum_exp, loss);
  //     for (c = 0; c < layer1_size; c++) grad[c] = 0;
  //     for (k = 0; k < nodes; k++) {
  //       if (tree[i].neg_ids[k] > 0 || i == k) {
  //         f = alpha * lambda_cat / tree[i].cur_size * (exp_list[k] / sum_exp - (i == k ? 1 : 0));
  //         for (c = 0; c < layer1_size; c++) 
  //           grad[c] -= f * (tree[k].emb[c] - log(exp_list[i]) * syn0[c + word*layer1_size]);
  //         for (c = 0; c < layer1_size; c++)
  //           tree[k].emb[c] -= f * (syn0[c + word*layer1_size] - log(exp_list[i]) * tree[k].emb[c]);
  //         g = 0;
  //         for (c = 0; c < layer1_size; c++) g += tree[k].emb[c] * tree[k].emb[c];
  //         for (c = 0; c < layer1_size; c++) tree[k].emb[c] /= sqrt(g);
  //       }
  //     }
  //     for (c = 0; c < layer1_size; c++)
  //       syn0[c + word*layer1_size] += grad[c];
  //     g = 0;
  //     for (c = 0; c < layer1_size; c++) g += syn0[c + word*layer1_size] * syn0[c + word*layer1_size];
  //     for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] /= sqrt(g);
  //     cnt += 1;
  //   }
  for (i = 1; i < nodes; i++)
    for (j = 0; j < tree[i].cur_size; j++) {
      word = tree[i].cur_words[j];
      f = 0;
      for (c = 0; c < layer1_size; c++) f += syn0[c + word*layer1_size] * tree[i].emb[c];
      // for (k = 0; k < nodes; k++) {
      //   if (tree[i].neg_ids[k] > 0 && i != k) {
      //     h = 0;
      //     for (c = 0; c < layer1_size; c++) h += syn0[c + word*layer1_size] * tree[k].emb[c];
      
      //     if (f - h < cat_margin) {
      //       loss += cat_margin - (f - h);
      //       // compute context word gradient
      //       for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      //       for (c = 0; c < layer1_size; c++) neu1e[c] += tree[i].emb[c] - f * syn0[c + word*layer1_size] 
      //                                                   + h * syn0[c + word*layer1_size] - tree[k].emb[c];
            
      //       // update positive tree node
      //       for (c = 0; c < layer1_size; c++) grad[c] = syn0[c + word*layer1_size] - f * tree[i].emb[c];
      //       for (c = 0; c < layer1_size; c++) tree[i].emb[c] += alpha * lambda_cat / tree[i].cur_size * grad[c];
      //       g = 0;
      //       for (c = 0; c < layer1_size; c++) g += tree[i].emb[c] * tree[i].emb[c];
      //       for (c = 0; c < layer1_size; c++) tree[i].emb[c] /= sqrt(g);

      //       // update negative tree node
      //       for (c = 0; c < layer1_size; c++) grad[c] = h * tree[k].emb[c] - syn0[c + word*layer1_size];
      //       for (c = 0; c < layer1_size; c++) tree[k].emb[c] += alpha * lambda_cat / tree[i].cur_size * grad[c];
      //       g = 0;
      //       for (c = 0; c < layer1_size; c++) g += tree[k].emb[c] * tree[k].emb[c];
      //       for (c = 0; c < layer1_size; c++) tree[k].emb[c] /= sqrt(g);

      //       // update context word
      //       for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] += alpha * lambda_cat / tree[i].cur_size * neu1e[c];
      //       g = 0;
      //       for (c = 0; c < layer1_size; c++) g += syn0[c + word*layer1_size] * syn0[c + word*layer1_size];
      //       for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] /= sqrt(g);
      //     }

      //   }
      // }
      if (f < cat_margin) {
        loss += cat_margin - f;
        for (c = 0; c < layer1_size; c++) grad[c] = syn0[c + word*layer1_size] - f * tree[i].emb[c];
        for (c = 0; c < layer1_size; c++) tree[i].emb[c] += alpha * lambda_cat / tree[i].cur_size * grad[c];
        g = 0;
        for (c = 0; c < layer1_size; c++) g += tree[i].emb[c] * tree[i].emb[c];
        for (c = 0; c < layer1_size; c++) tree[i].emb[c] /= sqrt(g);

        for (c = 0; c < layer1_size; c++) grad[c] = tree[i].emb[c] - f * syn0[c + word*layer1_size];
        for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] += alpha * lambda_cat / tree[i].cur_size * grad[c];
        g = 0;
        for (c = 0; c < layer1_size; c++) g += syn0[c + word*layer1_size] * syn0[c + word*layer1_size];
        for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] /= sqrt(g);
      }
      cnt += 1;
    }
  free(grad);
  free(neu1e);
  // free(exp_list);
  return loss / cnt;
}

void ExpandTopic() {
  long a, b, c;
  int cur_sz, flag;
  real norm;
  for (a = 1; a < nodes; a++) {
    for (b = 0; b < vocab_size; b++) {
      tree[a].wt_score[b] = 0;
      norm = 0.0;
      for (c = 0; c < layer1_size; c++) {
        tree[a].wt_score[b] += tree[a].emb[c] * syn0[b * layer1_size + c];
        norm += syn0[b * layer1_size + c] * syn0[b * layer1_size + c];
      }
      tree[a].wt_score[b] /= sqrt(norm);
    }
    wt_score_ptr = tree[a].wt_score;
    qsort(rankings, vocab_size, sizeof(int), SimCompare);
    cur_sz = tree[a].init_size;
    while (cur_sz < tree[a].cur_size + expand) {
      for (b = 0; b < vocab_size; b++) {
        flag = 0;
        for (c = 0; c < cur_sz; c++) {
          if (rankings[b] == tree[a].cur_words[c]) {
            flag = 1;
            break;
          }
        }
        if (flag == 0) {
          tree[a].cur_words[cur_sz++] = rankings[b];
          break;
        }
      }
    }
    tree[a].cur_size += expand;
    printf("Category: %s\t", tree[a].node_name);
    for (b = 0; b < tree[a].cur_size; b++) {
      printf("%s ", vocab[tree[a].cur_words[b]].word);
    }
    printf("\n");
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, doc = 0, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, l3 = 0, c, target, local_iter = 1;
  int word_counter = 0;
  unsigned long long next_random = (long long)id;
  char eof = 0;
  real f, g, h, step, tree_loss = 0, cat_loss = 0;
  clock_t now;
  real *neu1 = (real *) calloc(layer1_size, sizeof(real));
  real *grad = (real *) calloc(layer1_size, sizeof(real));
  real *neu1e = (real *) calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Tree Loss: %f  Category Loss: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 
         13, alpha, tree_loss, cat_loss,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      if (global_lambda > 0) doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi, &eof);
        if (eof) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    
    if (word_counter == tree_emb_period) word_counter = 0;
    if (word_counter == 0 && iter_count >= pretrain_iter) {
      tree_loss = TreeEmb();
      cat_loss = CatEmb();
    }
    word_counter++;

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size; // positive center word u
        
        // obj_w = 0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            l3 = word * layer1_size; // positive context word v
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            l2 = target * layer1_size; // negative center word u'
            f = 0;
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l3]; // f = cos(v, u) = v * u
            h = 0;
            for (c = 0; c < layer1_size; c++) h += syn0[c + l2] * syn1neg[c + l3]; // h = cos(v, u') = v * u'
        
            if (f - h < word_margin) {
              // obj_w += word_margin - (f - h);

              // compute context word gradient
              for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
              for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l1] - f * syn1neg[c + l3] + h * syn1neg[c + l3] - syn0[c + l2];
              
              // update positive center word
              for (c = 0; c < layer1_size; c++) grad[c] = syn1neg[c + l3] - f * syn0[c + l1]; // negative Riemannian gradient
              step = 1 - f; // cosine distance, d_cos
              for (c = 0; c < layer1_size; c++) syn0[c + l1] += alpha * step * grad[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn0[c + l1] * syn0[c + l1];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn0[c + l1] /= g;

              // update negative center word
              for (c = 0; c < layer1_size; c++) grad[c] = h * syn0[c + l2] - syn1neg[c + l3];
              step = 2 * h; // 2 * negative cosine similarity
              for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * step * grad[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn0[c + l2] * syn0[c + l2];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn0[c + l2] /= g;

              // update context word
              step = 1 - (f - h);
              for (c = 0; c < layer1_size; c++) syn1neg[c + l3] += alpha * step * neu1e[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn1neg[c + l3] * syn1neg[c + l3];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn1neg[c + l3] /= g;
            }
          }
        }
      }

    // obj_d = 0;
    l1 = doc * layer1_size; // positive document d
    for (d = 0; d < negative + 1; d++) {
      if (d == 0) {
        l3 = word * layer1_size; // positive center word u
      } else {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        target = table[(next_random >> 16) % table_size];
        if (target == 0) target = next_random % (vocab_size - 1) + 1;
        if (target == word) continue;
        l2 = target * layer1_size; // negative center word u'
      
        f = 0;
        for (c = 0; c < layer1_size; c++) f += syn0[c + l3] * syn1doc[c + l1]; // f = cos(u, d) = u * d
        h = 0;
        for (c = 0; c < layer1_size; c++) h += syn0[c + l2] * syn1doc[c + l1]; // h = cos(u', d) = u' * d
    
        if (f - h < word_margin) {
          // obj_d += word_margin - (f - h);

          // compute document gradient
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l3] - f * syn1doc[c + l1] + h * syn1doc[c + l1] - syn0[c + l2];

          // update positive center word
          for (c = 0; c < layer1_size; c++) grad[c] = syn1doc[c + l1] - f * syn0[c + l3];
          step = 1 - f;
          for (c = 0; c < layer1_size; c++) syn0[c + l3] += alpha * step * grad[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn0[c + l3] * syn0[c + l3];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn0[c + l3] /= g;

          // update negative center word
          for (c = 0; c < layer1_size; c++) grad[c] = h * syn0[c + l2] - syn1doc[c + l1];
          step = 2 * h;
          for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * step * grad[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn0[c + l2] * syn0[c + l2];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn0[c + l2] /= g;

          // update document
          step = 1 - (f - h);
          for (c = 0; c < layer1_size; c++) syn1doc[c + l1] += alpha * step * neu1e[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn1doc[c + l1] * syn1doc[c + l1];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn1doc[c + l1] /= g;
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(grad);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  InitUnigramTable();
  InitDocTable();
  start = clock();

  for (iter_count = 0; iter_count < iter; iter_count++) {
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    if (iter_count >= pretrain_iter) ExpandTopic();
  }

  fo = fopen(output_file, "wb");
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);

  fo = fopen(tree_emb_file, "wb");
  // Save the tree embedding vectors
  fprintf(fo, "%lld\n", nodes);
  for (a = 0; a < nodes; a++) {
    fprintf(fo, "%s ", tree[a].node_name);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&tree[a].emb[b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", tree[a].emb[b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-taxo-file", argc, argv)) > 0) strcpy(taxo_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-neg-file", argc, argv)) > 0) strcpy(neg_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-level-file", argc, argv)) > 0) strcpy(level_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-load-emb", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-global-lambda", argc, argv)) > 0) global_lambda = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda-tree", argc, argv)) > 0) lambda_tree = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda-cat", argc, argv)) > 0) lambda_cat = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-word-margin", argc, argv)) > 0) word_margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-tree-margin", argc, argv)) > 0) tree_margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-tree-margin-sub", argc, argv)) > 0) tree_margin_sub = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-cat-margin", argc, argv)) > 0) cat_margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tree-emb", argc, argv)) > 0) strcpy(tree_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-tree-period", argc, argv)) > 0) tree_emb_period = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-expand", argc, argv)) > 0) expand = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-pre-iter", argc, argv)) > 0) pretrain_iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  docs = (int *) calloc(corpus_max_size, sizeof(int));
  doc_sizes = (long long *)calloc(corpus_max_size, sizeof(long long));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
