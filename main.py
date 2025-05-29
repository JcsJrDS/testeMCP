from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

# Importações locais (estes arquivos seriam criados no mesmo diretório ou em subdiretórios)
import crud
import schemas
import genetic_scheduler
from database import SessionLocal, engine, Base

# Cria as tabelas no banco de dados (se não existirem)
# Em um cenário de produção, você poderia usar Alembic para migrações.
# Base.metadata.create_all(bind=engine) # Comentado para não executar automaticamente aqui

app = FastAPI(title="API de Planejamento de Produção")

# Configuração do CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Endereço do seu frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependência para obter a sessão do banco de dados
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Planejamento de Produção"}

@app.get("/operations_data/", response_model=List[schemas.OperationData])
def get_operations_data(
    cod_os: Optional[int] = Query(12474, description="Código da Ordem de Serviço para filtrar as operações."),
    db: Session = Depends(get_db)
):
    """
    Endpoint para buscar os dados brutos das operações do banco de dados
    com base na query fornecida.
    """
    try:
        operations = crud.get_operations_from_db(db, cod_os=cod_os)
        if not operations:
            raise HTTPException(status_code=404, detail=f"Nenhuma operação encontrada para a OS {cod_os}")
        return operations
    except Exception as e:
        # Em produção, logar o erro e retornar uma mensagem mais genérica se necessário
        raise HTTPException(status_code=500, detail=f"Erro ao buscar dados do banco: {str(e)}")


@app.post("/generate_schedule/", response_model=schemas.ProductionSchedule)
async def generate_production_schedule(
    planning_request: schemas.PlanningRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint para gerar o planejamento de produção otimizado.
    """
    try:
        # 1. Buscar dados das operações do banco
        operations_data = crud.get_operations_from_db(db, cod_os=planning_request.cod_os)
        if not operations_data:
            raise HTTPException(status_code=404, detail=f"Nenhuma operação encontrada para a OS {planning_request.cod_os} para planejamento.")

        # 2. Converter dados brutos para o formato esperado pelo algoritmo genético
        #    (Esta etapa pode envolver mais transformações dependendo da implementação do AG)
        initial_operations = [schemas.Operation(**op_data.dict(), id=f"{op_data.cod_os_completo}_{op_data.seq}") for op_data in operations_data]


        # 3. Executar o algoritmo genético
        #    Os parâmetros do AG (tamanho da população, gerações, etc.) podem vir da requisição
        #    ou serem configurados no backend.
        optimized_schedule_tasks, metrics = await genetic_scheduler.run_genetic_algorithm(
            operations=initial_operations,
            population_size=planning_request.population_size,
            num_generations=planning_request.num_generations,
            mutation_rate=planning_request.mutation_rate,
            crossover_rate=planning_request.crossover_rate,
            start_date_planning=planning_request.start_date # Passa a data de início do planejamento
        )
        
        # Formatar a resposta
        schedule_response = schemas.ProductionSchedule(
            cod_os=planning_request.cod_os,
            scheduled_operations=optimized_schedule_tasks,
            metrics=metrics # Inclui makespan, machine_utilization, etc.
        )
        
        return schedule_response

    except HTTPException as http_exc:
        raise http_exc # Re-lança exceções HTTP
    except Exception as e:
        # Logar o erro (e.g., com logging.error(str(e), exc_info=True))
        raise HTTPException(status_code=500, detail=f"Erro ao gerar o planejamento: {str(e)}")

# Para executar localmente: uvicorn main:app --reload
```
