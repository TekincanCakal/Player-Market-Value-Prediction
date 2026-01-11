import { NextResponse } from 'next/server';
import { db } from '@/utils/db'; // Correct import

export async function GET() {
    try {
        // db.query otomatik olarak havuzdan client alır, sorguyu yapar ve geri bırakır.
        const result = await db.query('SELECT * FROM players ORDER BY "Overall" DESC LIMIT 100');
        return NextResponse.json(result.rows);
    } catch (error) {
        console.error('Error fetching players:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
