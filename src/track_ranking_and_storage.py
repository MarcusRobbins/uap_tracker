import threading
import queue
import sqlite3
from .shared_variables import SharedVariables

class TrackRankingAndStorage:
    def __init__(self, shared_vars: SharedVariables):
        self.shared_vars = shared_vars
        self.connection = self.create_db_connection()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            try:
                if self.shared_vars.tracking_enabled:
                    # Get track from shared variables queue
                    track = self.shared_vars._tracking_queue.get_nowait()
                    
                    # Rank the track
                    ranked_track = self.rank_track(track)
                    
                    # Save to SQLite database
                    self.save_to_db(ranked_track)

                    # If the track's rank is above a certain threshold, add it to the 
                    # high_ranking_tracks queue in shared variables
                    if ranked_track['rank'] > self.shared_vars.ranking_threshold:
                        self.shared_vars.filtered_tracking_queue.put(ranked_track)

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Exception occurred: {e}")
                break

    def rank_track(self, track):
        # Here you would implement the logic for ranking the track
        # For example, you could add a 'rank' field to the track dictionary
        # track['rank'] = some_function(track)
        # return track
        pass

    def create_db_connection(self):
        try:
            connection = sqlite3.connect('yourdatabase.db')
            return connection
        except Exception as e:
            print("Error while connecting to SQLite", e)
            return None

    def save_to_db(self, ranked_track):
        if self.connection is not None:
            cursor = self.connection.cursor()
            add_track = ("INSERT INTO Tracks "
                         "(id, rank, data) "
                         "VALUES (?, ?, ?)")
            data_track = (ranked_track['id'], ranked_track['rank'], ranked_track['data'])
            cursor.execute(add_track, data_track)
            self.connection.commit()
            cursor.close()
        else:
            print("No connection to SQLite")
