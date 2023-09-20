'''
Created on Sep. 9, 2023

@author: cefect
'''
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, URL
from definitions import postgres_d, equal_area_epsg, postgres_dir

#===============================================================================
# POSTGRES--------
#===============================================================================
def get_conn_str(d):
    pg_str=''
    for k,v in d.items():
        pg_str+=f'{k}={v} ' 
        
    return pg_str[:-1]


def pg_vacuum(schema, tableName, conn_str=None):
    """perform vacuum and analyze on passed table
    
    does not work with context management"""
    
    if conn_str is None:conn_str = get_conn_str(postgres_d)
    
    conn = psycopg2.connect(conn_str)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    print(f"""VACUUM ANALYZE {schema}.{tableName}""")
    cur.execute(f"""VACUUM ANALYZE {schema}.{tableName}""")
    # Make the changes to the database persistent
    conn.commit()
    # Close communication with the database
    cur.close()
    conn.close()
    return conn, cur


def pg_spatialIndex(schema, tableName, columnName='geom', **kwargs):
    return pg_exe(f"""
                CREATE INDEX {tableName}_geom_idx
                    ON {schema}.{tableName}
                        USING GIST ({columnName});
                """, **kwargs)
 
            

def pg_exe(cmd_str, conn_str=None, log=None, return_fetch=False):
    if not log is None:
        log.info(cmd_str)
    if conn_str is None:conn_str = get_conn_str(postgres_d)
        
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(cmd_str)
            if return_fetch:
                return cur.fetchall()
 
        
def pg_getCRS(schema, tableName, geom_coln='geom', conn_str=None):
    """get the crs from a table"""
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
        
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT Find_SRID(%s, %s, %s)""", 
                        (schema, tableName, geom_coln)
                        )
            return int(cur.fetchone()[0])

def pg_getcount(schema, tableName,  conn_str=None):
    """get the crs from a table"""
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
        
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT COUNT(*) FROM {schema}.{tableName}""")
            return int(cur.fetchone()[0])

def pg_to_df(cmd_str, conn_d=postgres_d):
    """load a filtered table to geopanbdas"""
    
    conn =  psycopg2.connect(get_conn_str(conn_d))
    #set engine for geopandas
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    try:
        result = pd.read_sql_query(cmd_str, engine)
        
    except Exception as e:
        raise IOError(f'failed query w/ \n    {e}')
    finally:
        # Dispose the engine to close all connections
        engine.dispose()
        # Close the connection
        conn.close()
        

    return result


def pg_get_column_names(schema, tableName, conn_str=None):
    """get the column names""" 
    
    if conn_str is None:conn_str = get_conn_str(postgres_d)
        
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT column_name
                    FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s
                        ORDER BY ordinal_position
                        """,(tableName, schema))
 
            l = cur.fetchall()
    assert len(l)>0, f'table {schema}.{tableName} does not exist'
            
    return [e[0] for e in l]

def pg_register(schema, tableName, conn_str=None):
    """register the geometry of the table"""
    if conn_str is None:conn_str = get_conn_str(postgres_d)
        
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT Populate_Geometry_Columns(%s::regclass)""", (f'{schema}.{tableName}', ))
            
def pg_comment(schema, tableName, cmt_str, conn_str=None):
    """alter the comment on a table
    
    cmt_str = f'port of {cnt} .gpkg sample files on grid centroids\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%S")
    
    """
    if conn_str is None:conn_str = get_conn_str(postgres_d)
        
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""COMMENT ON TABLE {schema}.{tableName} IS %s""", (cmt_str, ))
            
            
            
            
            
            
            
            