class MusicDatabase:

    def __init__(self, path=PATH):
        self.connection = sqlite3.connect(PATH)

    @contextmanager
    def transaction(self):
        if not self.connection:
            raise RuntimeError("No connection established")

        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    @property
    def playlist_count(self):
        with self.transaction() as cursor:
            cursor.execute("SELECT COUNT(*) FROM playlists")
            count = cursor.fetchone()[0]

        return count

    def get_playlist_audio(self, playlist_id):
        """Get file paths for the audio of all tracks in the playlist.

        Args:
            playlist_id (int): playlist id.

        Returns:
            audio_paths (dict[track_id (str), file_path (str)]): key-value pairs of the corresponding file paths for track ids.
        """
        command, parameters = read.playlist_tracks_id_and_audio_path(playlist_id)

        with self.transaction() as cursor:
            cursor.execute(command, parameters)
            data = cursor.fetchall()

        audio_paths = {}

        for track_id, audio_path in data:
            audio_paths[track_id] = audio_path

        return audio_paths
