def playlist_tracks_id_and_audio_path(playlist_id):
    command = """
        SELECT t.id, t.audio_path
        FROM tracks t
        JOIN playlist_tracks pt ON t.id = pt.track_id
        WHERE pt.pid = ?
        AND t.audio_path IS NOT NULL;
    """
    parameters = (playlist_id,)
    return command, parameters
