CREATE TABLE urls (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    status VARCHAR(20) CHECK (status IN ('legitimate', 'phishing')) NOT NULL
);
