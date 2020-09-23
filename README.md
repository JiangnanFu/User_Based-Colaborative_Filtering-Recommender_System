Go through this [Report](https://docs.google.com/document/d/1_HuEZYfmOBlCWAlKkjJDSJCP7Ccf1UAYeFLHAHG9omo/edit?usp=sharing) to get details of  concepts used.

**Input**

  *   Given the ratings file(ratings.csv) of  1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.
  *   Column Format : - UserID::MovieID::Rating::Timestamp
  *   UserIDs range between 1 and 6040 
  *   MovieIDs range between 1 and 3952
  *   Ratings are made on a 5-star scale (whole-star ratings only)
  *   Timestamp is represented in seconds since the epoch as returned by time(2)
  *   Each user has at least 20 ratings

**OutPut**

  Predict the ratings for a movie for the Users who didn’t watch that movie.

**Technical Description**

  *   Get the required data from here [data set](https://grouplens.org/datasets/movielens/100k/).
  *   File description 
      *   **100KNHSM.py** : - The code predicts the ratings based on User similarity and uses New Heuristic similarity Measure(NHSM) for computing the similarity.  To Test the                               results data are divided into training and test Sets.
      *   **AC_NHSM_5Fold.py** : - This gives the rating prediction by combining the idea of User similarity along with item similarity. To compute similarity between Two                                          items(Movie) Adjudicated Cosine(AC) Distance  is used, whereas for user similarity computation NHSM were used. Here to validate for more                                        accuracy , data are divided into 5 fold training sets and test sets.
  *   Code can be executed either through any ide built in run module(i.e. spyder) or using command line , provided data should present beforehand.
  *   We set a Threshold after computing the rating of a Movie for Particular users as 4, If the predicted rating is below 4 , we don’t recommend that movie to users.
