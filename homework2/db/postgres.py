
import psycopg

class PostgresRepository:

    def __init__(self, connection_string: str):

        self.connection = psycopg.connect(connection_string)

        # Register pgvector support for this connection
        

    def insert_chunks(self, embedded_chunks: list[dict]):

        query = """
            INSERT INTO document_chunks
                (filename, chunk_id, content, embedding)
            VALUES
                (%s, %s, %s, %s)
            ON CONFLICT (filename, chunk_id)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding;
        """

        data = [
            (
                chunk["filename"],
                chunk["chunkId"],
                chunk["content"],
                chunk["embedding"]
            )
            for chunk in embedded_chunks
        ]

        with self.connection.cursor() as cursor:
            cursor.executemany(query, data)

        self.connection.commit()

        print("Successfully stored:", len(data))

    def close(self):
        self.connection.close()
