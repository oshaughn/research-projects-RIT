#!/usr/bin/env python
import sys
import math

import sqlite3

# Stolen from gstlal_inspiral_plotsummary
sim_coinc_map = """
CREATE TEMPORARY TABLE
    sim_coinc_map
AS
    SELECT
        sim_inspiral.simulation_id AS simulation_id,
        (
            SELECT
                coinc_inspiral.coinc_event_id
            FROM
                coinc_event_map AS a
                JOIN coinc_event_map AS b ON (
                    b.coinc_event_id == a.coinc_event_id
                )
                JOIN coinc_inspiral ON (
                    b.table_name == 'coinc_event'
                    AND b.event_id == coinc_inspiral.coinc_event_id
                )
            WHERE
                a.table_name == 'sim_inspiral'
                AND a.event_id == sim_inspiral.simulation_id
            ORDER BY
                coinc_inspiral.false_alarm_rate
            LIMIT 1
        ) AS coinc_event_id,
        (
            SELECT
                coinc_inspiral.end_time
            FROM
                coinc_event_map AS a
                JOIN coinc_event_map AS b ON (
                    b.coinc_event_id == a.coinc_event_id
                )
                JOIN coinc_inspiral ON (
                    b.table_name == 'coinc_event'
                    AND b.event_id == coinc_inspiral.coinc_event_id
                )
            WHERE
                a.table_name == 'sim_inspiral'
                AND a.event_id == sim_inspiral.simulation_id
            ORDER BY
                coinc_inspiral.false_alarm_rate
            LIMIT 1
        ) AS coinc_end_time,
        (
            SELECT
                coinc_inspiral.end_time_ns
            FROM
                coinc_event_map AS a
                JOIN coinc_event_map AS b ON (
                    b.coinc_event_id == a.coinc_event_id
                )
                JOIN coinc_inspiral ON (
                    b.table_name == 'coinc_event'
                    AND b.event_id == coinc_inspiral.coinc_event_id
                )
            WHERE
                a.table_name == 'sim_inspiral'
                AND a.event_id == sim_inspiral.simulation_id
            ORDER BY
                coinc_inspiral.false_alarm_rate
            LIMIT 1
        ) AS coinc_end_time_ns
    FROM
        sim_inspiral
    WHERE
        coinc_event_id IS NOT NULL
"""

select_coincs = """
SELECT
    -- sim_inspiral.*,
    -- sngl_inspiral.*
    sim_coinc_map.coinc_event_id, sngl_inspiral.ifo, sngl_inspiral.mass1, sngl_inspiral.mass2, sngl_inspiral.snr, sim_coinc_map.coinc_end_time, sim_coinc_map.coinc_end_time_ns
FROM
    sim_inspiral
    JOIN sim_coinc_map ON (
        sim_coinc_map.simulation_id == sim_inspiral.simulation_id
    )
    JOIN coinc_event_map ON (
        coinc_event_map.coinc_event_id == sim_coinc_map.coinc_event_id
    )
    JOIN sngl_inspiral ON (
        coinc_event_map.table_name == 'sngl_inspiral'
        AND coinc_event_map.event_id == sngl_inspiral.event_id
    )
    WHERE sngl_inspiral.snr > 4.0 and sim_inspiral.simulation_id == "sim_inspiral:simulation_id:%d"
""" 

count_coincs = """
SELECT COUNT(*) FROM sim_coinc_map
"""

def count_sim_coinc(db):
    return int(list(db.execute(count_coincs))[0][0])

def get_coinc(db, sim_id):
    net_snr = 0
    result = list(db.execute(select_coincs % sim_id))
    # Already ordered by FAR take lowest FAR
    for id, ifo, m1, m2, snr, et, et_ns in result[:2]:
        net_snr += snr**2
    return id, m1, m2, et, et_ns, math.sqrt(net_snr)

def add_tmp_table(db):
    db.execute(sim_coinc_map)

if __file__ == sys.argv[0]:
    connection = sqlite3.connect(sys.argv[1])
    add_tmp_table(connection)
    id, m1, m2, etime, etime_ns, net_snr = get_coinc(connection, int(sys.argv[2]))
    connection.close()
    print id, m1, m2, "%d.%d" % (etime, etime_ns), net_snr 
