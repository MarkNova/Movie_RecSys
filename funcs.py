NAME = 'funcs'

class PopularRecommender():

    def __init__(self,
                 max_K=100,
                 days=30,
                 item_column='item_id',
                 dt_column='date',
                 user_column='user_id',
                 parallel=False):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.user_column = user_column
        self.dt_column = dt_column
        self.parallel = parallel
        self.recommendations = []

    def fit(
        self,
        df,
    ):
        self.df = df
        self.users = df[self.user_column].unique()

        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(
            days=self.days)

        self.recommendations = df.loc[df[self.dt_column] > min_date,
                                      self.item_column].value_counts().head(
                                          self.max_K).index.values

    def recommend(self,
                  users=None,
                  N=10,
                  one_person=False,
                  person_movie_list=None):
        if users is None:
            return self.recommendations[:N]

        if isinstance(users, np.ndarray):
            users = pd.Series(users)
            if self.parallel:
                recs = users.parallel_apply(lambda x: np.setdiff1d(
                    self.recommendations, self.df[self.df[
                        self.user_column] == x][self.item_column].values)[:N]
                                            if x in self.users else self.
                                            recommendations[:N].tolist())
            else:
                recs = users.progress_apply(lambda x: np.setdiff1d(
                    self.recommendations, self.df[self.df[
                        self.user_column] == x][self.item_column].values)[:N]
                                            if x in self.users else self.
                                            recommendations[:N].tolist())

            recs = pd.DataFrame(recs)
            recs[Columns.User] = users
            recs.columns = [Columns.Item, Columns.User]
            recs = recs.explode(column='item_id', ignore_index=True)

            ranks = []
            for i in recs['user_id'].unique():
                n = len(recs[recs['user_id'] == i]['item_id'])
                ranks += [j for j in range(1, n + 1)]
            recs[Columns.Rank] = ranks

            return recs

def validation_func(interactions=None,
                    cv=None,
                    model=None,
                    metrics=None,
                    k_recs=None):
    split_results = pd.DataFrame({key: [] for key in metrics.keys()})

    for train_ind, test_ind, _ in tqdm(cv.split(interactions)):
        train_df = interactions.df.loc[train_ind, :]
        df_test = interactions.df.loc[test_ind, :]
        users_test = df_test[Columns.User].unique()

        model.fit(train_df)

        recommendations = model.recommend(users=users_test, N=k_recs)

        results = calc_metrics(metrics,
                               reco=recommendations,
                               interactions=df_test,
                               prev_interactions=train_df,
                               catalog=train_df[Columns.Item].unique())
        results = pd.DataFrame(results, index=[0])

        split_results = pd.concat([split_results, results], ignore_index=True)

    split_results['model'] = 'PopularModel'
    del model

    return split_results.groupby('model').agg([np.mean, np.std])