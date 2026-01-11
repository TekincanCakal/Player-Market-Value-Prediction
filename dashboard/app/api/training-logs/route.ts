import { NextResponse } from 'next/server';
import { db } from '@/utils/db';

export const dynamic = 'force-dynamic'; // Her istekte veriyi taze çek

export async function GET() {
    try {
        // Son 1 eğitim kaydını getir
        const { rows } = await db.sql`
      SELECT * FROM training_logs 
      ORDER BY timestamp DESC 
      LIMIT 1;
    `;

        // Geçmiş 50 kaydın özetini getir (Grafik için)
        const history = await db.sql`
        SELECT timestamp, mae_score, final_loss 
        FROM training_logs 
        ORDER BY timestamp ASC 
        LIMIT 50;
    `;

        return NextResponse.json({
            latest: rows[0] || null,
            history: history.rows
        });

    } catch (error) {
        console.error('Database Error:', error);
        return NextResponse.json({ error: 'Failed to fetch data' }, { status: 500 });
    }
}
