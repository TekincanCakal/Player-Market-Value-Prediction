import { NextResponse } from 'next/server';
import { db } from '@/utils/db';

export const dynamic = 'force-dynamic'; // Her istekte veriyi taze çek

export async function GET() {
    try {
        // Son 1 eğitim kaydını getir
        const latestResult = await db.query(`
      SELECT * FROM training_logs 
      ORDER BY timestamp DESC 
      LIMIT 1;
    `);

        // Geçmiş 50 kaydın özetini getir (Grafik için)
        const historyResult = await db.query(`
        SELECT timestamp, mae_score, final_loss 
        FROM training_logs 
        ORDER BY timestamp ASC 
        LIMIT 50;
    `);

        return NextResponse.json({
            latest: latestResult.rows[0] || null,
            history: historyResult.rows
        });

    } catch (error) {
        console.error('Database Error:', error);
        return NextResponse.json({ error: 'Failed to fetch data' }, { status: 500 });
    }
}
