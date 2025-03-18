
CREATE TABLE tracks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    audio_path TEXT
);

CREATE TABLE playlists (
    pid INTEGER PRIMARY KEY,
    name TEXT
);

CREATE TABLE playlist_tracks (
    pid INTEGER,
    track_id TEXT,
    FOREIGN KEY (pid) REFERENCES playlists (pid),
    FOREIGN KEY (track_id) REFERENCES tracks (id),
    PRIMARY KEY (pid, track_id)
);

-- The tables below are not relevant for this use case but would be of future interest
/*
CREATE TABLE artists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE albums (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    album_image_path TEXT
);

CREATE TABLE track_albums (
    track_id TEXT,
    album_id TEXT,
    FOREIGN KEY (track_id) REFERENCES tracks (id),
    FOREIGN KEY (album_id) REFERENCES albums (id),
    PRIMARY KEY (track_id, album_id)
);

CREATE TABLE album_artists (
    album_id TEXT,
    artist_id TEXT,
    FOREIGN KEY (album_id) REFERENCES albums (id),
    FOREIGN KEY (artist_id) REFERENCES artists (id),
    PRIMARY KEY (album_id, artist_id)
);

CREATE TABLE track_artists (
    track_id TEXT,
    artist_id TEXT,
    FOREIGN KEY (track_id) REFERENCES tracks (id),
    FOREIGN KEY (artist_id) REFERENCES artists (id),
    PRIMARY KEY (track_id, artist_id)
);

CREATE TABLE playlist_track_orders (
    pid INTEGER,
    track_id TEXT,
    pos INTEGER,
    FOREIGN KEY (pid) REFERENCES playlists (pid),
    FOREIGN KEY (track_id) REFERENCES tracks (id),
    PRIMARY KEY (pid, pos)
);
*/