# iteratively build vocab for a company over all articles
def build_vocab(raw_articles, stopwordset):
    onegrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    # iterate over dates and paragraphs
    for key in raw_articles.keys():
        for paragraph in raw_articles[key].values():
            # reset variables
            word = ''
            biword = []
            triword = []
            # cast paragraph into string
            string = str(paragraph)
            # iterate over characters
            for char in string:
                if char.isalpha():
                    word += char
                else:
                    # evaluate current word candidate if hit a nonalpha character
                    if word and word not in stopwordset:
                        # add to biword and triword
                        biword.append(word)
                        triword.append(word)
                        # update grams
                        onegrams[word] += 1   
                        if len(biword) == 2:
                            bigrams[biword[0] + ' ' + biword[1]] += 1
                            biword.pop(0)
                        if len(triword) == 3:
                            trigrams[triword[0] + ' ' + triword[1] + ' ' + triword[2]] += 1
                            triword.pop(0)
                    # next word
                    word = ''
    return onegrams, bigrams, trigrams

# tokenize words in an article
def tokenize(raw_articles, stopwordset, onegrams, bigrams, trigrams, hdata_copy):    
    # construct feature data frames with empty spots for word grams and populate
    df_onegram = hdata_copy.copy()
    df_bigram = hdata_copy.copy()
    df_trigram = hdata_copy.copy()
    print('onegrams...')
    for gram in tqdm(onegrams):
        df_onegram[gram] = 0
    print('bigrams...')
    for gram in tqdm(bigrams):
        df_bigram[gram] = 0
    print('trigrams...')
    for gram in tqdm(trigrams):
        df_trigram[gram] = 0
    # iterate over dates and paragraphs
    print('building dataframe...')
    for key in tqdm(raw_articles.keys()):
        if not any(hdata_copy['DATE']==key): continue # don't continue if no stock price for the date
        for paragraph in raw_articles[key].values():
            # reset variables
            word = ''
            biword = []
            triword = []
            # cast paragraph into string
            string = str(paragraph)
            # iterate over characters
            for char in string:
                if char.isalpha():
                    word += char
                else:
                    # evaluate current word candidate if hit a nonalpha character
                    if word and word not in stopwordset:
                        # add to biword and triword
                        biword.append(word)
                        triword.append(word)
                        # update dataframes
                        if word in onegrams:
                            df_onegram.at[np.where(df_onegram['DATE']==key)[0][0], word] += 1                
                        if len(biword) == 2:
                            bistr = biword[0] + ' ' + biword[1]
                            biword.pop(0)
                            if bistr in bigrams:
                                df_bigram.at[np.where(df_bigram['DATE']==key)[0][0], bistr] += 1
                        if len(triword) == 3:
                            tristr = triword[0] + ' ' + triword[1] + ' ' + triword[2]
                            triword.pop(0)
                            if tristr in trigrams:
                                df_trigram.at[np.where(df_trigram['DATE']==key)[0][0], tristr] += 1
                    # next word
                    word = ''
    # project vocab to lower dimension
    pca = PCA(n_components=200, svd_solver='full')
    df_onegram = np.transpose(pca.fit_transform(df_onegram.drop(columns=['DATE', 'ADJ_CLOSE'])))
    print('onegram variance explained:', sum(pca.explained_variance_ratio_)) 
    pca = PCA(n_components=200, svd_solver='full')
    df_bigram = np.transpose(pca.fit_transform(df_bigram.drop(columns=['DATE', 'ADJ_CLOSE'])))
    print('bigram variance explained:', sum(pca.explained_variance_ratio_)) 
    pca = PCA(n_components=200, svd_solver='full')
    df_trigram = np.transpose(pca.fit_transform(df_trigram.drop(columns=['DATE', 'ADJ_CLOSE'])))
    print('trigram variance explained:', sum(pca.explained_variance_ratio_)) 
    return df_onegram, df_bigram, df_trigram

# return tokenized words under all setups
def generate_df(name, hdata):
    # stopword set
    stopwordset = set(stopwords.words('english'))
    # open file with pickle (as binary)
    f = open('news_data/' + name + '.pickle', 'rb')
    raw_articles = pickle.load(f)
    f.close()
    # build vocab
    print('building vocab...')
    onegrams, bigrams, trigrams = build_vocab(raw_articles, stopwordset)
    # chop vocab
    # onegrams = onegrams.most_common(300)
    bigrams = bigrams.most_common(len(onegrams))
    trigrams = trigrams.most_common(len(onegrams))
    # onegrams = dict((x, y) for x,y in onegrams)
    bigrams = dict((x, y) for x,y in bigrams)
    trigrams = dict((x, y) for x,y in trigrams)
    # populate feature data frame with tokenized words
    print(name, len(onegrams), len(bigrams), len(trigrams))
    print('tokenizing...')
    df_onegram, df_bigram, df_trigram = tokenize(raw_articles, stopwordset, onegrams, bigrams, trigrams, hdata.copy())
    # reconstruct feature matrix
    feature_df_onegram = hdata.copy()
    for i in range(len(df_onegram)):
        feature_df_onegram['PCA_' + str(i)] = df_onegram[i]
    feature_df_bigram = hdata.copy()
    for i in range(len(df_bigram)):
        feature_df_bigram['PCA_' + str(i)] = df_bigram[i]
    feature_df_trigram = hdata.copy()
    for i in range(len(df_trigram)):
        feature_df_trigram['PCA_' + str(i)] = df_trigram[i]
    return feature_df_onegram, feature_df_bigram, feature_df_trigram