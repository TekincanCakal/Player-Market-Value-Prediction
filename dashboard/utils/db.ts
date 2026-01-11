import { Pool } from 'pg';

export const db = new Pool({
    connectionString: process.env.POSTGRES_URL || 'postgresql://admin:admin@localhost:5432/football_db',
});

// Vercel Postgres ile uyumluluk için 'sql' wrapper (Eski kodları bozmamak adına)
// Eski kodda: await db.sql`SELECT * FROM ...`
// Yeni kodda: db.query(...) şeklinde olmalı ama
// geçiş kolay olsun diye bir wrapper yazıyoruz veya direkt route'ları güncelleyeceğiz.
// En temiz yöntem route'ları pg syntax'ına geçirmek.
