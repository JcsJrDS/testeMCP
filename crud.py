from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import schemas

def get_operations_from_db(db: Session, cod_os: Optional[int] = 12474) -> List[schemas.OperationData]:
    """
    Executa a consulta SQL fornecida para buscar os dados das operações.
    """
    # A query original fornecida pelo usuário
    # ATENÇÃO: É crucial validar/sanitizar `cod_os` se ele for diretamente na query
    # para evitar SQL Injection. Usar parâmetros de query é mais seguro.
    query_str = """
        SELECT tpro_pro.cod_os_completo, tpro_pro.seq, tpro_pro.cod_maquina,
               tpro_pro.total_hrs, tpro_pro.hs_realizadas, tpro_pro.maquina,
               tpro_pro.cod_os, tos.dt_prevista, tos.prioridade_preactor,
               tpro_pro.cod_serv_terceiro
        FROM tpro_pro
        INNER JOIN tos ON tpro_pro.cod_os = tos.codigo
        WHERE (tpro_pro.cod_os = :cod_os_param) AND (tpro_pro.finalizado = 0)
        ORDER BY tpro_pro.seq
    """
    # Executa a query usando text() para queries SQL literais e passando parâmetros
    result = db.execute(text(query_str), {"cod_os_param": cod_os})
    
    # Mapeia o resultado para o schema Pydantic
    # Os nomes das colunas no SELECT devem corresponder aos campos em OperationData
    operations = []
    for row in result:
        # O SQLAlchemy 2.0 retorna Row objects que podem ser acessados por nome ou índice
        # Para mapear para Pydantic, é mais fácil converter para um dict primeiro
        # ou garantir que o schema Pydantic possa lidar com o Row object (com orm_mode=True)
        op_data = {
            "cod_os_completo": row.cod_os_completo,
            "seq": row.seq,
            "cod_maquina": row.cod_maquina,
            "total_hrs": row.total_hrs,
            "hs_realizadas": row.hs_realizadas,
            "maquina": row.maquina,
            "cod_os": row.cod_os,
            "dt_prevista": row.dt_prevista,
            "prioridade_preactor": row.prioridade_preactor,
            "cod_serv_terceiro": row.cod_serv_terceiro,
        }
        operations.append(schemas.OperationData(**op_data))
    return operations
