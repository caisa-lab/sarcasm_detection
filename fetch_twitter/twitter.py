import csv
import re
import tweepy


class Twitter:
    def __init__(self, consumer_key, consumer_secret, access_token, access_secret):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self._api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        
    @staticmethod
    def collect_users(text):
        return re.findall(r'@(\w{1,15})', text) 
    
    def user_data(self, id):
        try:
            return self._api.get_user(id).id_str
        except tweepy.TweepError as e:
            if e.api_code == 50:
                return 'Error: User Not Found'
            if e.api_code == 63:
                return 'Error: User Suspended'
            raise e

    def lookup(self, ids, batch_size):
        """response object will be empty except for the id itself."""
        for batch in (ids[i:i + batch_size] for i in range(0, len(ids), batch_size)):
            for status in self._api.statuses_lookup(batch, tweet_mode='extended', map=True):
                yield status

    def lookup_file(self, file_path, delimiter='\t', id_index=0, header=True):
        with open(file_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            if header:
                next(reader)
            data = list(reader)  # this loads the file into memory, but i need to iterate it twice, so there's that.
        for meta, tweet in zip(data, self.lookup([line[id_index] for line in data])):
            text = re.sub(r'\s+', ' ', getattr(tweet, 'full_text', '').replace('\n', ' '))
            user = tweet.user.id_str if hasattr(tweet, 'user') else ''
            yield meta + [user, text]

    def lookup_user(self, user, count=200):
        """queries a user's timeline for his or her post history. Because this method uses tweepy's Cursor, the
        'count' parameter applies to a single request, while the method will potentially send multiple requests in
        order to request up to 3200 tweets for the given user id:
        https://developer.twitter.com/en/docs/twitter-api/v1/tweets/timelines/api-reference/get-statuses-user_timeline"""
        for status in tweepy.Cursor(self._api.user_timeline, id=user, tweet_mode='extended', count=count).items():
            yield status