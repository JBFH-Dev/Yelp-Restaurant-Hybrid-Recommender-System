# -*- coding: utf-8 -*-

# Used in the calculation of evaluation metrics
import random
import statistics
# Used in making the UI print at a readable speed
import time

# Used to store mathematically efficient data in matrices
import numpy as np
# Used to hold data and parse it
import pandas as pd
# Used to create sparse pivot table
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix
# Used in generating TF-IDF similarity matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
# Used to time testing loops
from tqdm import tqdm


# Function for setting up similarity matrices and pivot table
def setup(reviews):
    # Creates ordered list of user ids
    users = sorted(reviews.user_id.unique())
    # Creates ordered list of business ids
    businesses = sorted(reviews.business_id.unique())
    #
    # Method for creating sparse pivot table adapted from:
    # https://stackoverflow.com/questions/31661604/efficiently-create-sparse-pivot-tables-in-pandas
    #
    user_c = CategoricalDtype(users, ordered=True)
    business_c = CategoricalDtype(businesses, ordered=True)
    row = reviews.user_id.astype(user_c).cat.codes
    col = reviews.business_id.astype(business_c).cat.codes
    # Creates sparse matrix populated by ratings with columns of businesses and rows of users
    sparse_matrix = csr_matrix((reviews["stars"], (row, col)),
                               shape=(user_c.categories.size, business_c.categories.size))
    # Deletes unnecessary variables to save memory
    del reviews
    del col
    del row
    del user_c
    del business_c
    # del businesses
    # Converts dtype of sparse matrix to float32 to massively reduce memory usage
    sparse_matrix = sparse_matrix.astype(np.float32)
    # Calculates average ratings made by each user (row-wise mean)
    #
    # non-zero numpy divide found on:
    # https://stackoverflow.com/questions/38542548/numpy-mean-of-nonzero-values
    #
    #
    averages = np.true_divide(sparse_matrix.sum(1), (sparse_matrix != 0).sum(1))
    # Calculates cosine similarity of items and creates sparse matrix of results
    collab_similarities = cosine_similarity(sparse_matrix.transpose(), dense_output=False)
    # Loads the businesses df from a custom JSON
    businessesdf = pd.read_json('Data/yelp_dataset/yelp_academic_dataset_business_filt.json',
                                lines=True).sort_values(by=['business_id'])
    justStrings = businessesdf.drop(columns=['address', 'postal_code', 'is_open', 'hours', 'name'])
    # Deletes businesses dataframe to conserve memory
    del businessesdf
    #
    #
    # Similarity matrix generation for content-based filtering based on:
    # https://heartbeat.fritz.ai/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831
    #
    #
    # creates vectorizer in sklearn to recognise English unicode and ngrams of length 1-3 words long
    tf = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 2), stop_words='english', min_df=0)
    # fits the vectorizer to the categories data
    tfidf_matrix = tf.fit_transform(justStrings['categories'])
    # generates cosine similarity between all businesses
    content_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    # sets diagonal of similarity matrix to 0 as opposed to 1
    for i in range(content_similarities.shape[0]):
        content_similarities[i][i] = 0.0
    return sparse_matrix, averages, collab_similarities, csr_matrix(content_similarities), users, businesses


# Function for finding index of user id in matrix
def new_who(users):
    # Asks for a user id until a valid one is entered
    while True:
        print("Hi there, thanks for using Yelp")
        user = input("What's your user ID? ")
        if user not in users:
            print("Sorry that user ID isn't in our system, try again")
            continue
        else:
            # returns index of active user in sparse matrix
            user_no = users.index(user)
            return user_no


# Function for performing weighted average given similarity matrix and rated items
def recommender(sparse_matrix, similarities_sparse, user):
    user_rated = np.nonzero(sparse_matrix[user, ...])[1]
    # A is the matrix of similarities between all items and those rated by user
    A1 = similarities_sparse[user_rated, :]
    all_cols = np.arange(A1.shape[1])
    cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, user_rated.astype(int).tolist())))[0]
    A = A1[:, cols_to_keep]
    # Calculates denominator of prediction function through column-wise sum
    Denom = np.sum(A, axis=0)
    # R is the rating given by the user to each item
    R = sparse_matrix[user, user_rated].transpose()
    # Multiplying A by R
    B = A.multiply(R)
    # Calculates numerator for prediction function
    Numer = np.sum(B, axis=0)
    # Divides numerator matrix by denominator matrix where denominator != 0 to avoid NaN
    P = np.squeeze(np.asarray(np.divide(Numer, Denom, out=np.zeros_like(Numer), where=Denom != 0)))
    # Selects top 5 items based on prediction and returns them alongside their prediction
    recommendations = P.argsort()
    predictions = [P[i] for i in recommendations][::-1]
    return recommendations, predictions, user_rated


# Function for creating dataframe from predictions
def make_df(businesses, scores):
    predicted_scores = pd.DataFrame(
        {'business_id': businesses,
         'predicted_rating': scores
         }).sort_values(by=['predicted_rating'], ascending=False)
    return predicted_scores


# Function for using the content-based predictions to break ties in the collaborative
def tie_breaker(collab_table, content_table):
    # pointer used to find number of tying items
    pointer = 0
    # moves pointer down list until no longer tying
    while pointer < collab_table.shape[0] - 1:
        ties = []
        value = collab_table.loc[pointer].predicted_rating.item()
        selected = collab_table.loc[pointer].business_id
        ties.append(selected)
        # if the following item is not a tie, continue
        if collab_table.loc[pointer + 1].predicted_rating.item() != value:
            pointer += 1
            continue
        i = 1
        # add all tying businesses to list
        while (collab_table.loc[pointer + i].predicted_rating.item() == value) and (
                i + pointer < collab_table.shape[0] - 1):
            ties.append(collab_table.loc[pointer + i].business_id)
            i += 1
        breakers = []
        # iterate through tying businesses, adding content-based scores to list
        for b in ties:
            breakers.append(content_table.loc[content_table.business_id == b].predicted_rating.item())
        # find range of content-based scores in order to normalise and break ties
        break_min = min(breakers)
        break_range = max(breakers) - break_min
        if break_range != 0:
            breakers = [((n - break_min) / break_range) for n in breakers]
        else:
            # if all ties also tie in content-based then simply add 0.5 to each
            breakers = [0.5 for n in breakers]
        for i in range(len(ties)):
            # for each tie, add the tiebreaker to the predicted score
            collab_table.loc[i + pointer, 'predicted_rating'] += breakers[i]
        pointer += len(ties)
    collab_table.sort_values(by=['predicted_rating'], ascending=False, inplace=True)
    # the below code was added to normalise the final predicted score to be in the correct range
    # however this made RMSE worse
    #
    # max_pred = collab_table['predicted_rating'].max()
    # min_pred = collab_table['predicted_rating'].min()
    # pred_range = max_pred - min_pred
    # func = lambda x: (5*(x-min_pred))/pred_range
    # collab_table['predicted_rating'] = collab_table['predicted_rating'].map(func)
    return collab_table


# Function for removing 20* of user's reviews
def review_remover(user_id, reviews):
    # get reviews left by user
    user_rated = reviews[reviews['user_id'] == user_id].reset_index()
    num_rated = user_rated.shape[0]
    # if only one review was left, skip user
    if num_rated == 1:
        return -100
    # find number to remove
    remove_num = round(num_rated * 0.2)
    rated_indices = [i for i in range(num_rated)]
    random.shuffle(rated_indices)
    # select random indices to remove
    remove_indices = [rated_indices[j] for j in range(remove_num)]
    removed_reviews = user_rated.loc[remove_indices]
    removed_businesses = removed_reviews['business_id'].tolist()
    # remove selected reviews
    reviews = reviews[~((reviews['business_id'].isin(removed_businesses)) & (reviews['user_id'] == user_id))]
    return (reviews, removed_reviews)


# runs recommender with no user id requested
def hybrid_filter_noinput(reviews, user):
    sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews)
    # retrieves indices of rated items by the active user
    user = users.index(user)
    recommendations_col, predictions_col, user_rated = recommender(sparse_matrix, collab_similarities, user)
    recommendations_con, predictions_con, x = recommender(sparse_matrix, content_similarities, user)
    del x
    businesses = [businesses[j] for j in range(len(businesses)) if j not in user_rated.tolist()]
    businesses_col = [businesses[i] for i in recommendations_col]
    businesses_con = [businesses[i] for i in recommendations_con]
    return predictions_col, predictions_con, businesses_col, businesses_con


# runs entire program with no user requested
def run_noinput(reviews, user):
    predictions_col, predictions_con, businesses_col, businesses_con = hybrid_filter_noinput(reviews, user)
    col_table = make_df(businesses_col, predictions_col)
    con_table = make_df(businesses_con, predictions_con)
    col_table = tie_breaker(col_table, con_table)
    return col_table.reset_index().drop(columns=['index'])


# retrieves top 500 active users
def get_top_users(reviews, percent):
    reviews = reviews.drop(columns=['business_id'])
    reviews = reviews.groupby(['user_id']).count().reset_index()
    reviews = reviews.sort_values(by=['stars'], ascending=False)
    best_users = reviews['user_id'].tolist()
    best_users = best_users[:500]  # len(best_users) // (100 // percent)]
    return best_users


# finds difference between predicted and actual ratings
def comparison_pred(col_table, removed_reviews):
    removed_businesses = removed_reviews['business_id'].tolist()
    real_ratings = []
    predicted_ratings = []
    # for each hidden review, find predicted value
    for b in removed_businesses:
        pred_rating = col_table[col_table['business_id'] == b].predicted_rating.item()
        real_rating = removed_reviews[removed_reviews['business_id'] == b].stars.item()
        real_ratings.append(real_rating)
        predicted_ratings.append(pred_rating)
    return real_ratings, predicted_ratings


# runs entire evaluation testing suite
def all_stats(num_of_recs=5):
    print("- - - Loading Reviews - - -")
    # Loads the filtered reviews df from a custom JSON consisting of just user, business and stars
    reviews = pd.read_json('Data/yelp_dataset/yelp_academic_dataset_review_filt.json',
                           lines=True).drop(columns=['date', 'review_id'])
    print("- - - Loaded - - -")
    ratings = reviews.groupby(['business_id', 'user_id']).mean()
    reviews = ratings.reset_index()
    # gets users to be evaluated
    users = get_top_users(reviews, 1)
    RMSE = []
    precision_l = []
    diversity_l = []
    # for each user calculate all three metrics
    for u in tqdm(users):
        reviews_removed_reviews = review_remover(u, reviews)
        if reviews_removed_reviews == -100:
            continue
        reviews_short = reviews_removed_reviews[0]
        removed_reviews = reviews_removed_reviews[1]
        # predict ratings on test set
        col_table = run_noinput(reviews_short, u)
        recommended_items = col_table.head(num_of_recs).business_id.tolist()
        # DIVERSITY
        # see report for mathematical details
        sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews_short)
        sims = []
        for i in recommended_items:
            for j in recommended_items:
                index_i = businesses.index(i)
                index_j = businesses.index(j)
                sim = collab_similarities[index_i, index_j]
                neg_sim = 1 - sim
                sims.append(neg_sim)
        Num = sum(sims)
        N = len(recommended_items)
        Den = (N / 2) * (N - 1)
        diversity_l.append(Num / Den)
        # PRECISION
        removed_businesses = removed_reviews['business_id'].tolist()
        TP = list(set(removed_businesses) & set(recommended_items))
        FP = [i for i in recommended_items if (i not in removed_businesses)]
        precision = len(TP) / (len(TP) + len(FP))
        precision_l.append(precision)
        # RMSE
        real_ratings, predicted_ratings = comparison_pred(col_table, removed_reviews)
        all_real = np.asarray(real_ratings)
        all_pred = np.asarray(predicted_ratings)
        rmse = np.sqrt(np.mean(np.subtract(all_pred, all_real) ** 2))
        RMSE.append(rmse)
    return statistics.mean(RMSE), statistics.mean(precision_l), statistics.mean(diversity_l)


# prints the menu
def frontend():
    time.sleep(0.5)
    print('----------------------------------------')
    time.sleep(0.5)
    print('----------------------------------------')
    time.sleep(0.5)
    print('----------------------------------------')
    time.sleep(0.5)
    print('-----------------YELP-------------------')
    time.sleep(0.5)
    print('------------ACADEMIC DATASET------------')
    time.sleep(0.5)
    print('--                                    --')
    time.sleep(0.5)
    print('----MENU--------------------------------')
    time.sleep(0.5)
    print('-- 1. Settings                        --')
    time.sleep(0.5)
    print('-- 2. Get Recommendations             --')
    time.sleep(0.5)
    print('-- 3. See Your Ratings                --')
    time.sleep(0.5)
    print('-- 4. Quit                            --')
    time.sleep(0.5)
    print()
    user_response = '0'
    while (not user_response.isnumeric()) or (int(user_response) < 1) or (int(user_response) > 4):
        user_response = input('-- ')
    return int(user_response)


# shows user the settings and allows changes to be made
def settings(explainability_mode, include_covid):
    print('--                                    --')
    time.sleep(0.5)
    print('----SETTINGS----------------------------')
    time.sleep(0.5)
    print('-- 1. Number of Recommendations       --')
    time.sleep(0.5)
    print('-- 2. Toggle explainability mode:     --')
    time.sleep(0.5)
    print('-------- Currently set to: ', explainability_mode)
    time.sleep(0.5)
    print('-- 3. Toggle include_covid:           --')
    time.sleep(0.5)
    print('-------- Currently set to: ', include_covid)
    time.sleep(0.5)
    user_response = '0'
    while (not user_response.isnumeric()) or (int(user_response) < 1) or (int(user_response) > 3):
        user_response = input('-- ')
    if int(user_response) == 1:
        user_response = '0'
        while (int(user_response) > 1455206) or (int(user_response) < 1) or (not user_response.isnumeric()):
            user_response = input('-- Number:')
        return int(user_response)
    if int(user_response) == 2:
        explainability_mode = not explainability_mode
        print('-------- Explainability set to: ', explainability_mode)
        time.sleep(0.5)
        return -1
    if int(user_response) == 3:
        include_covid = not include_covid
        print('-------- Include_covid set to: ', include_covid)
        time.sleep(0.5)
        return -2


# shows items rated by the user
def see_ratings(reviews):
    while True:
        print()
        time.sleep(0.5)
        user = input("-- What's your user ID? ")
        if user not in reviews['user_id'].tolist():
            time.sleep(0.5)
            print("-- Sorry that user ID isn't in our system, try again")
            continue
        else:
            break
    ratings = reviews.set_index(['user_id']).loc[user]
    ratings = merge_business_data(ratings)
    print(ratings)


# runs recommender in explainability mode, describing each step to the user
def explainability(reviews, num_recs, user):
    print("- - - Loading Reviews - - -")
    sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews)
    print("- - - Loaded - - -")
    time.sleep(0.5)
    print('-- Firstly, matrices are generated consisting of the similarities between all the businesses')
    time.sleep(0.5)
    print('-- The first of these uses similarities based on how similar users rated them')
    time.sleep(0.5)
    print('-- The second uses similarities based on the descriptions of the businesses')
    time.sleep(0.5)
    recommendations_col, predictions_col, user_rated = recommender(sparse_matrix, collab_similarities, user)
    recommendations_con, predictions_con, x = recommender(sparse_matrix, content_similarities, user)
    print(
        '-- Then two recommenders are run, using these similarities to predict how you would rate each unseen business')
    time.sleep(0.5)
    del x
    businesses_short = [businesses[j] for j in range(len(businesses)) if j not in user_rated.tolist()]
    businesses_col = [businesses_short[i] for i in recommendations_col]
    businesses_con = [businesses_short[i] for i in recommendations_con]
    col_table = make_df(businesses_col, predictions_col)
    con_table = make_df(businesses_con, predictions_con)
    col_table = tie_breaker(col_table, con_table).head(num_recs)
    col_table = merge_business_data(col_table)
    print('-- Then, businesses are sorted by what we predict you will rate them')
    time.sleep(0.5)
    print(
        '-- And any businesses that have the same predicted rating in the first recommender, are ordered by the second')
    time.sleep(0.5)
    print('-- Finally the top predictions are printed as recommendations:')
    time.sleep(0.5)
    print(col_table)


# runs all evaluation metrics on just collaborative filter
def all_stats_baseline(num_of_recs=5):
    print("- - - Loading Reviews - - -")
    # Loads the filtered reviews df from a custom JSON consisting of just user, business and stars
    reviews = pd.read_json('Data/yelp_dataset/yelp_academic_dataset_review_filt.json',
                           lines=True).drop(columns=['date', 'review_id'])
    print("- - - Loaded - - -")
    ratings = reviews.groupby(['business_id', 'user_id']).mean()
    reviews = ratings.reset_index()
    users = get_top_users(reviews, 1)
    RMSE = []
    precision_l = []
    diversity_l = []
    for u in tqdm(users):
        reviews_removed_reviews = review_remover(u, reviews)
        if reviews_removed_reviews == -100:
            continue
        reviews_short = reviews_removed_reviews[0]
        removed_reviews = reviews_removed_reviews[1]
        col_table = run_noinput_baseline(reviews_short, u)
        recommended_items = col_table.head(num_of_recs).business_id.tolist()
        # DIVERSITY
        sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews_short)
        sims = []
        for i in recommended_items:
            for j in recommended_items:
                index_i = businesses.index(i)
                index_j = businesses.index(j)
                sim = collab_similarities[index_i, index_j]
                neg_sim = 1 - sim
                sims.append(neg_sim)
        Num = sum(sims)
        N = len(recommended_items)
        Den = (N / 2) * (N - 1)
        diversity_l.append(Num / Den)
        # PRECISION
        removed_businesses = removed_reviews['business_id'].tolist()
        TP = list(set(removed_businesses) & set(recommended_items))
        FP = [i for i in recommended_items if (not i in removed_businesses)]
        precision = len(TP) / (len(TP) + len(FP))
        precision_l.append(precision)
        # RMSE
        real_ratings, predicted_ratings = comparison_pred(col_table, removed_reviews)
        all_real = np.asarray(real_ratings)
        all_pred = np.asarray(predicted_ratings)
        rmse = np.sqrt(np.mean(np.subtract(all_pred, all_real) ** 2))
        RMSE.append(rmse)
    return statistics.mean(RMSE), statistics.mean(precision_l), statistics.mean(diversity_l)


# runs full collaborative recommender
def run_noinput_baseline(reviews, user):
    predictions_col, predictions_con, businesses_col, businesses_con = hybrid_filter_noinput(reviews, user)
    col_table = make_df(businesses_col, predictions_col)
    return col_table.reset_index().drop(columns=['index'])


# generates the business_ids of businesses currently open or doing takeaway/delivery
def generate_covid_allowed():
    cd = pd.read_json('Data/covid_19_dataset_2020_06_10/yelp_academic_dataset_covid_filt.json',
                      lines=True)
    cd_businesses = cd['business_id'].tolist()
    return cd_businesses


# merges business info with predictions
def merge_business_data(recommended):
    businesses = pd.read_json('Data/yelp_dataset/yelp_academic_dataset_business_filt.json',
                              lines=True).drop(
        columns=['is_open', 'attributes', 'categories', 'address', 'postal_code', 'hours'])
    cd = pd.read_json('Data/covid_19_dataset_2020_06_10/yelp_academic_dataset_covid_filt.json',
                      lines=True)
    cd = cd.loc[cd['business_id'].isin(recommended['business_id'].tolist())]
    businesses = businesses.loc[businesses['business_id'].isin(recommended['business_id'].tolist())].rename(
        columns={"stars": "avg_stars"})
    merged = recommended.merge(businesses, how='inner', on='business_id')
    merged = merged.merge(cd, how='inner', on='business_id').drop(columns=['business_id'])
    columns = merged.columns.tolist()
    columns = [columns[1]] + [columns[0]] + columns[2:]
    merged = merged[columns]
    del businesses
    del cd
    return merged


# runs full evaluation suite on both baseline and hybrid at varying k
def run_evaluation():
    print('Hybrid: 5')
    print(all_stats(5))
    print()
    print('Collab: 5')
    print(all_stats_baseline(5))
    print()
    print('Hybrid: 30')
    print(all_stats(30))
    print()
    print('Collab: 30')
    print(all_stats_baseline(30))


# sets pandas to display all columns
pd.set_option('display.max_columns', None)
print("- - - Loading Reviews - - -")
# Loads the filtered reviews df from a custom JSON consisting of just user, business and stars
reviews = pd.read_json('Data/yelp_dataset/yelp_academic_dataset_review_filt.json',
                       lines=True).drop(columns=['date', 'review_id'])
# ensures each user has only reviewed any restaurant once
ratings = reviews.groupby(['business_id', 'user_id']).mean()
reviews = ratings.reset_index()
del ratings
# pre-calculates similarities
sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews)
print("- - - Loaded - - -")
num_recs = 5
explainability_mode = False
include_covid = True
# shows user menus
while True:
    user_response = frontend()
    if user_response == 4:
        quit()
    if user_response == 1:
        setting = settings(explainability_mode, include_covid)
        if setting == -1:
            explainability_mode = not explainability_mode
        elif setting == -2:
            include_covid = not include_covid
            if not include_covid:
                reviews = reviews.loc[reviews['business_id'].isin(generate_covid_allowed())]
                sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews)
            else:
                print("- - - Loading Reviews - - -")
                # Loads the filtered reviews df from a custom JSON consisting of just user, business and stars
                reviews = pd.read_json('Data/yelp_dataset/yelp_academic_dataset_review_filt.json',
                                       lines=True).drop(columns=['date', 'review_id'])
                ratings = reviews.groupby(['business_id', 'user_id']).mean()
                reviews = ratings.reset_index()
                del ratings
                sparse_matrix, averages, collab_similarities, content_similarities, users, businesses = setup(reviews)
                print("- - - Loaded - - -")
        else:
            num_recs = setting
    elif user_response == 3:
        see_ratings(reviews)
        print()
        input("-- Press Enter to continue..")
    elif user_response == 2:
        user = new_who(users)
        if user == -1:
            print('quitting')
            quit()
        if explainability_mode:
            col_table = explainability(reviews, num_recs, user)
            print()
            input('-- Press Enter to continue..')
            continue
        # runs recommender and displays results to user
        recommendations_col, predictions_col, user_rated = recommender(sparse_matrix, collab_similarities, user)
        recommendations_con, predictions_con, x = recommender(sparse_matrix, content_similarities, user)
        del x
        businesses_short = [businesses[j] for j in range(len(businesses)) if j not in user_rated.tolist()]
        businesses_col = [businesses_short[i] for i in recommendations_col]
        businesses_con = [businesses_short[i] for i in recommendations_con]
        col_table = make_df(businesses_col, predictions_col)
        con_table = make_df(businesses_con, predictions_con)
        col_table = tie_breaker(col_table, con_table).head(num_recs)
        col_table = merge_business_data(col_table)
        print(col_table)
        time.sleep(2)

"""ISSUE:

--- Also would like to adjust similarity measures using averages and possibly remove sparsity

TEST USER:

ofKDkJKXSKZXu5xJNGiiBQ

TEST RESULT:

Ae91-YlrGnHKq7KXD1WDxw	6.000000

QWrSZtFD8vRVeXZjInRZ9A	5.930669

hLu4q64lPvF--9gd58lcPg	5.930669

AmIJMMd9OeT_mUj9DAn2ow	5.901977

QXPjl1G5XxlJhc8vaH2NLg  5.900251


OTHER TEST USER IDS:
Here are some user_ids you can use:

ofKDkJKXSKZXu5xJNGiiBQ
bLbSNkLggFnqwNNzzq-Ijw
PKEzKWv_FktMm2mGPjwd0Q
UYcmGbelzRa0Q6JqzLoguw
U4INQZOPSUaj8hMjLlZ3KA
C2C0GPKvzWWnP57Os9eQ0w
n86B7IkbU20AkxlFX_5aew
8DEyKVyplnOcSKx39vatbg
3nDUQBjKyVor5wV0reJChg
N3oNEwh0qgPqPP3Em6wJXw
tH0uKD-vNwMoEc3Xk3Cbdg
qewG3X2O4X6JKskxyyqFwQ
JaqcCU3nxReTW2cBLHounA
B1829_hxXSEpDPEDJtYeIw
L8P5OWO1Jh4B2HLa1Fnbng
_VMGbmIeK71rQGwOBWt_Kg
YE54kKTuqJJPNYWIKIpOEQ
iSC96O2NjQc3JExGUHQG0Q
y3FcL4bLy0eLlkb0SDPnBQ
oeAhRa8yFa9jtrhaHnOyxQ
48vRThjhuhiSQINQ2KV8Sw
qPVtjjp8sNQ32p9860SR9Q





ADJUSTED TO 0-5, 500 USERS
Hybrid: 5
(1.179752843488058, 0.0, 1.999669847032428)


"""
