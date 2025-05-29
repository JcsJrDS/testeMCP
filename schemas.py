from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, date

class OperationBase(BaseModel):
    cod_os_completo: str
    seq: int
    cod_maquina: Optional[str] = None
    maquina: Optional[str] = None
    total_hrs: float
    hs_realizadas: float
    cod_serv_terceiro: int
    dt_prevista: date # Vem da tabela 'tos'
    prioridade_preactor: Optional[int] = None # Vem da tabela 'tos'
    cod_os: int # Vem da tabela 'tos', usado para join

class OperationData(OperationBase): # Schema para dados vindos do DB
    class Config:
        orm_mode = True

class Operation(OperationBase): # Schema usado internamente pelo AG
    id: str # Identificador único para a operação (pode ser cod_os_completo + seq)
    # Campos adicionais calculados pelo AG
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "Pendente" # Pendente, Em Andamento, Concluído

class PlanningRequest(BaseModel):
    cod_os: int
    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    start_date: Optional[date] = date.today() # Data de início do planejamento

class ScheduleMetrics(BaseModel):
    makespan_hours: Optional[float] = None # Tempo total para completar todas as ops
    average_machine_utilization: Optional[float] = None # %
    total_tardiness_days: Optional[float] = None # Atraso total em relação às datas previstas

class ProductionSchedule(BaseModel):
    cod_os: int
    scheduled_operations: List[Operation]
    metrics: Optional[ScheduleMetrics] = None
