'''
Created on Sep. 9, 2023

@author: cefect
'''

import psycopg2
from definitions import postgres_d, equal_area_epsg, postgres_dir

#===============================================================================
# POSTGRES--------
#===============================================================================
def get_conn_str(d):
    pg_str=''
    for k,v in d.items():
        pg_str+=f'{k}={v} ' 
        
    return pg_str[:-1]


def pg_vacuum(conn_d, tableName):
    """perform vacuum and analyze on passed table
    
    does not work with context management"""
    conn = psycopg2.connect(get_conn_str(conn_d))
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(f"""VACUUM ANALYZE {tableName}""")
    # Make the changes to the database persistent
    conn.commit()
    # Close communication with the database
    cur.close()
    conn.close()
    return conn, cur


def pg_spatialIndex(conn_d, schema, tableName, columnName='geom'):
    return pg_exe(f"""
                CREATE INDEX {tableName}_geom_idx
                    ON {schema}.{tableName}
                        USING GIST ({columnName});
                """)
 
            

def pg_exe(cmd_str, conn_d=None, log=None, return_fetch=False):
    if not log is None:
        log.info(cmd_str)
    if conn_d is None:
        conn_d =postgres_d
        
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            cur.execute(cmd_str)
            if return_fetch:
                return cur.fetchall()
 
        
def pg_getCRS(schema, tableName, geom_coln='geom', conn_d=None):
    """get the crs from a table"""
    
    if conn_d is None:
        conn_d =postgres_d
        
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT Find_SRID(%s, %s, %s)""", 
                        (schema, tableName, geom_coln)
                        )
            return int(cur.fetchone()[0])
    
 