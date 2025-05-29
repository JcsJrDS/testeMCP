import random
from typing import List, Tuple, Dict
from datetime import datetime, timedelta, date
import schemas # Importa os schemas Pydantic
from utils_datetime import (
    adicionar_horas_uteis, 
    adicionar_dias_uteis, 
    eh_dia_util, 
    INICIO_TURNO, 
    HORAS_TRABALHO_POR_DIA,
    calculate_total_working_hours # Import the new helper function
)
import utils_datetime # Added to resolve proximo_dia_util not defined

# Representação de uma tarefa agendada (para uso interno no AG e resultado)
class ScheduledTask(schemas.Operation):
    start_time: datetime
    end_time: datetime
    machine_id: str # cod_maquina

# Função para calcular o tempo de processamento restante
def get_processing_time_hours(op: schemas.Operation) -> float:
    return op.total_hrs - op.hs_realizadas

async def run_genetic_algorithm(
    operations: List[schemas.Operation],
    population_size: int,
    num_generations: int,
    mutation_rate: float,
    crossover_rate: float,
    start_date_planning: date = date.today()
) -> Tuple[List[ScheduledTask], schemas.ScheduleMetrics]:
    """
    Executa o algoritmo genético para otimizar o planejamento.
    Retorna a lista de tarefas agendadas e métricas do planejamento.
    Esta é uma implementação ESQUELETO e precisa ser detalhada.
    """
    print(f"Iniciando AG com {len(operations)} operações. Pop: {population_size}, Gen: {num_generations}")

    if not operations:
        return [], schemas.ScheduleMetrics()

    # Garante que a data de início do planejamento seja um dia útil e tenha hora de início do turno
    current_planning_datetime = datetime.combine(start_date_planning, INICIO_TURNO)
    if not eh_dia_util(current_planning_datetime.date()):
        current_planning_datetime = datetime.combine(
            utils_datetime.proximo_dia_util(current_planning_datetime.date()), 
            INICIO_TURNO
        )

    # 1. Inicializar População
    # Um indivíduo é uma permutação da ordem das operações.
    # O decodificador irá alocar as operações às máquinas.
    population = [random.sample(operations, len(operations)) for _ in range(population_size)]

    best_overall_schedule = None
    best_overall_fitness = -float('inf')
    best_overall_metrics = schemas.ScheduleMetrics() # Initialize with default metrics

    for generation in range(num_generations):
        print(f"Geração {generation + 1}/{num_generations}")
        fitness_scores = []
        schedules_this_generation = []

        for i, individual_ops_order in enumerate(population):
            # 2. Decodificar Indivíduo para um Planejamento (Schedule)
            # Esta é a parte mais complexa: alocar operações às máquinas respeitando restrições.
            # (sequência, uma op por máquina, turnos, serviços de terceiros)
            current_schedule_tasks, schedule_metrics = decode_individual_to_schedule(
                individual_ops_order, current_planning_datetime
            )
            schedules_this_generation.append(current_schedule_tasks)
            
            # 3. Calcular Fitness
            # O fitness deve considerar: makespan, atrasos (dt_prevista), utilização de máquinas, prioridade_preactor.
            fitness = calculate_fitness(current_schedule_tasks, schedule_metrics, operations)
            fitness_scores.append(fitness)

            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_schedule = current_schedule_tasks
                best_overall_metrics = schedule_metrics
                print(f"  Novo melhor fitness na geração {generation+1}, indivíduo {i}: {fitness:.2f}, Makespan: {schedule_metrics.makespan_hours:.2f}h")


        if not best_overall_schedule: # Caso nenhuma solução válida seja encontrada
            print("Nenhuma solução válida encontrada nas primeiras tentativas.")
            if schedules_this_generation and schedules_this_generation[0]:
                best_overall_schedule = schedules_this_generation[0]
                # Ensure population is not empty before accessing its elements
                if population:
                    best_overall_metrics = decode_individual_to_schedule(population[0], current_planning_datetime)[1]
                else: # Fallback if population is empty
                    best_overall_metrics = schemas.ScheduleMetrics(makespan_hours=0, average_machine_utilization=0, total_tardiness_days=0)

            else: # Se nem isso, retorna vazio
                return [], schemas.ScheduleMetrics(makespan_hours=0, average_machine_utilization=0, total_tardiness_days=0)


        # 4. Seleção (Ex: Roleta, Torneio)
        selected_parents = selection_tournament(population, fitness_scores, num_parents=population_size)

        # 5. Crossover (Ex: Um Ponto, Dois Pontos, Order Crossover - OX1 para permutações)
        # 6. Mutação (Ex: Swap, Scramble para permutações)
        next_population = []
        if selected_parents: # Ensure selected_parents is not empty
            for i in range(0, len(selected_parents), 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i+1] if (i+1) < len(selected_parents) else selected_parents[i] 
                
                child1, child2 = parent1[:], parent2[:] 

                if random.random() < crossover_rate:
                    child1, child2 = ordered_crossover(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child1 = swap_mutation(child1)
                if random.random() < mutation_rate:
                    child2 = swap_mutation(child2)
                
                next_population.extend([child1, child2])
        
        population = next_population[:population_size] 
        if not population and selected_parents: 
            population = selected_parents[:population_size]
        elif not population and not selected_parents and operations: 
            population = [random.sample(operations, len(operations)) for _ in range(population_size)]


    print("Algoritmo Genético concluído.")
    if not best_overall_schedule:
        print("AVISO: Nenhum planejamento ótimo foi encontrado. Retornando vazio.")
        return [], schemas.ScheduleMetrics()

    return best_overall_schedule, best_overall_metrics

def decode_individual_to_schedule(
    ordered_operations: List[schemas.Operation],
    initial_start_datetime: datetime
) -> Tuple[List[ScheduledTask], schemas.ScheduleMetrics]:
    """
    Decodifica uma ordem de operações (indivíduo) em um planejamento detalhado.
    Aloca operações a máquinas e calcula tempos de início/fim.
    Esta é a função crítica que implementa as regras de agendamento.
    """
    scheduled_tasks: List[ScheduledTask] = []
    machine_finish_times: Dict[str, datetime] = {} 
    operation_finish_times: Dict[str, datetime] = {} 
    
    ops_by_project: Dict[str, List[schemas.Operation]] = {}
    for op in ordered_operations: 
        ops_by_project.setdefault(op.cod_os_completo, []).append(op)

    for proj_code in ops_by_project:
        ops_by_project[proj_code].sort(key=lambda o: o.seq)

    processed_op_ids = set()
    temp_ordered_ops = [] 

    all_ops_map = {f"{op.cod_os_completo}_{op.seq}": op for op in ordered_operations}

    for op_from_individual in ordered_operations:
        op_id = f"{op_from_individual.cod_os_completo}_{op_from_individual.seq}"
        
        predecessor_finish_time = initial_start_datetime
        if op_from_individual.seq > 1:
            prev_op_id = f"{op_from_individual.cod_os_completo}_{op_from_individual.seq - 1}"
            if prev_op_id in operation_finish_times:
                predecessor_finish_time = operation_finish_times[prev_op_id]
                # If predecessor not found (should not happen in a well-formed OS),
                # fallback to initial_start_datetime for this op's predecessor time.
                # This might indicate an issue with OS data or ordering.
                else: 
                    predecessor_finish_time = initial_start_datetime

        processing_hrs = get_processing_time_hours(op_from_individual)
        op_start_time_for_schedule = initial_start_datetime # Default initialization
        op_end_time = initial_start_datetime # Default initialization

        if processing_hrs <= 0:
            # Adjust predecessor_finish_time to a valid slot if it's not.
            # If processing_hrs is 0, start and end times are the same, adjusted.
            op_start_time_for_schedule = adicionar_horas_uteis(predecessor_finish_time, 0)
            op_end_time = op_start_time_for_schedule
            
            # Update machine finish time if a machine is assigned, even for zero-hour ops,
            # as it might occupy the machine momentarily or represent a setup.
            if op_from_individual.cod_maquina:
                 machine_finish_times[op_from_individual.cod_maquina] = max(
                    machine_finish_times.get(op_from_individual.cod_maquina, initial_start_datetime),
                    op_end_time # which is same as op_start_time_for_schedule here
                )
            operation_finish_times[op_id] = op_end_time
            scheduled_tasks.append(ScheduledTask(
                **op_from_individual.dict(),
                id=op_id,
                start_time=op_start_time,
                end_time=op_end_time,
                machine_id=op_from_individual.cod_maquina if op_from_individual.cod_maquina else "TERCEIRO",
                status="Concluído" if op_from_individual.hs_realizadas >= op_from_individual.total_hrs else "Pendente"
            ))
            continue

        if op_from_individual.cod_serv_terceiro == 1:
            dias_terceiro = int(op_from_individual.total_hrs) # Assumes total_hrs for terceiros is in days
            
            # Adjust predecessor_finish_time to the start of the next available working slot
            base_start_time_for_terceiro = adicionar_horas_uteis(predecessor_finish_time, 0)

            # Third-party services are assumed to start at INICIO_TURNO.
            # If base_start_time_for_terceiro is already INICIO_TURNO, this is fine.
            # If it's later in the day (e.g. 10:00), the service starts on the next day's INICIO_TURNO.
            op_start_date_candidate = base_start_time_for_terceiro.date()
            if base_start_time_for_terceiro.time() > INICIO_TURNO:
                op_start_date_candidate = utils_datetime.proximo_dia_util(base_start_time_for_terceiro.date())
            
            op_start_time_for_schedule = datetime.combine(op_start_date_candidate, INICIO_TURNO)
            
            # Ensure the calculated start date is a working day (proximo_dia_util should handle this)
            if not eh_dia_util(op_start_time_for_schedule.date()):
                 op_start_time_for_schedule = datetime.combine(utils_datetime.proximo_dia_util(op_start_time_for_schedule.date()), INICIO_TURNO)

            op_end_date = adicionar_dias_uteis(op_start_time_for_schedule.date(), dias_terceiro)
            op_end_time = datetime.combine(op_end_date, INICIO_TURNO) # Finishes at the start of the day after duration.
            
            machine_id_for_task = "TERCEIRO_" + str(op_from_individual.cod_os_completo) + "_" + str(op_from_individual.seq)

        else: # Regular machine operation
            if not op_from_individual.cod_maquina:
                # This case should ideally be handled: what if a non-terceiro op has no machine?
                # For now, skip it or raise an error. Skipping for robustness.
                print(f"AVISO: Operação {op_id} não é de terceiro mas não tem máquina. Pulando.")
                continue 
            
            machine_id_for_task = op_from_individual.cod_maquina
            earliest_machine_available_time = machine_finish_times.get(machine_id_for_task, initial_start_datetime)
            
            # Tentative start time is the max of predecessor finishing or machine availability
            tentative_op_start_time = max(predecessor_finish_time, earliest_machine_available_time)

            # Adjust tentative_op_start_time to the actual start of a valid working slot.
            # The `adicionar_horas_uteis` function with 0 hours will perform this adjustment.
            actual_op_start_time = adicionar_horas_uteis(tentative_op_start_time, 0)
            
            # Now calculate the end time based on the correctly adjusted start time and processing hours
            op_end_time = adicionar_horas_uteis(actual_op_start_time, processing_hrs)
            
            op_start_time_for_schedule = actual_op_start_time
            machine_finish_times[machine_id_for_task] = op_end_time

        operation_finish_times[op_id] = op_end_time
        scheduled_tasks.append(ScheduledTask(
            **op_from_individual.dict(),
            id=op_id,
            start_time=op_start_time_for_schedule, # Use the correctly determined start time
            end_time=op_end_time,
            machine_id=machine_id_for_task,
            status="Pendente" 
        ))

    makespan_hours = 0
    if operation_finish_times:
        latest_finish_time = max(operation_finish_times.values()) if operation_finish_times else initial_start_datetime
        # total_work_seconds = (latest_finish_time - initial_start_datetime).total_seconds() # Not directly used for makespan_hours
        makespan_hours = calculate_total_working_hours(initial_start_datetime, latest_finish_time) # Use helper for makespan in working hours
    
    # Calculate Average Machine Utilization
    machine_actual_processing_time: Dict[str, float] = {}
    machine_first_op_start_time: Dict[str, datetime] = {}
    machine_last_op_end_time: Dict[str, datetime] = {}

    for task in scheduled_tasks:
        if not task.machine_id.startswith("TERCEIRO") and task.machine_id:
            # Actual processing time for the operation (original hours to be done)
            original_op = all_ops_map.get(task.id)
            if original_op:
                task_processing_hours = get_processing_time_hours(original_op) # Use original op for this
                machine_actual_processing_time[task.machine_id] = \
                    machine_actual_processing_time.get(task.machine_id, 0) + task_processing_hours
                
                if task.machine_id not in machine_first_op_start_time or task.start_time < machine_first_op_start_time[task.machine_id]:
                    machine_first_op_start_time[task.machine_id] = task.start_time
                if task.machine_id not in machine_last_op_end_time or task.end_time > machine_last_op_end_time[task.machine_id]:
                    machine_last_op_end_time[task.machine_id] = task.end_time
    
    machine_utilizations: List[float] = []
    for machine_id, actual_work_done_hrs in machine_actual_processing_time.items():
        if machine_id in machine_first_op_start_time and machine_id in machine_last_op_end_time:
            first_start = machine_first_op_start_time[machine_id]
            last_end = machine_last_op_end_time[machine_id]
            
            if last_end > first_start: # Ensure there's a duration
                available_hours_for_machine = utils_datetime.calculate_total_working_hours(first_start, last_end)
                if available_hours_for_machine > 0:
                    utilization = (actual_work_done_hrs / available_hours_for_machine) * 100
                    machine_utilizations.append(min(utilization, 100.0)) # Cap at 100%
    
    avg_machine_utilization = 0
    if machine_utilizations:
        avg_machine_utilization = sum(machine_utilizations) / len(machine_utilizations)
    else: # Handle cases with no machine operations or zero available hours
        avg_machine_utilization = 0 # Or some other appropriate default

    total_tardiness_days = 0
    for task in scheduled_tasks:
        original_op = all_ops_map.get(task.id)
        if original_op and original_op.dt_prevista:
            tardiness = (task.end_time.date() - original_op.dt_prevista).days
            if tardiness > 0:
                total_tardiness_days += tardiness
    
    metrics = schemas.ScheduleMetrics(
        makespan_hours=round(makespan_hours, 2),
        average_machine_utilization=round(avg_machine_utilization,2), 
        total_tardiness_days=round(total_tardiness_days,2)
    )
    return scheduled_tasks, metrics


def calculate_fitness(schedule: List[ScheduledTask], metrics: schemas.ScheduleMetrics, base_operations: List[schemas.Operation]) -> float:
    """
    Calcula a pontuação de fitness para um planejamento.
    Quanto maior, melhor.
    Objetivos: Minimizar makespan, minimizar atrasos, maximizar utilização, respeitar prioridades.
    """
    if not schedule: 
        return -float('inf')

    fitness_makespan = 1.0 / (1.0 + metrics.makespan_hours) if metrics.makespan_hours is not None else 0
    fitness_tardiness = 1.0 / (1.0 + metrics.total_tardiness_days) if metrics.total_tardiness_days is not None else 0
    fitness_utilization = (metrics.average_machine_utilization / 100.0) if metrics.average_machine_utilization is not None else 0
    
    priority_penalty = 0
    for task in schedule:
        original_op = next((op for op in base_operations if f"{op.cod_os_completo}_{op.seq}" == task.id), None)
        if original_op and original_op.dt_prevista and original_op.prioridade_preactor is not None:
            if task.end_time.date() > original_op.dt_prevista:
                priority_penalty += (1.0 / (original_op.prioridade_preactor + 1e-6)) * (task.end_time.date() - original_op.dt_prevista).days
    
    fitness_priority = 1.0 / (1.0 + priority_penalty)

    w_makespan = 0.35 # Adjusted
    w_tardiness = 0.3  # Kept
    w_utilization = 0.15 # Increased from 0.1 to 0.15
    w_priority = 0.2   # Kept
    
    total_fitness = (w_makespan * fitness_makespan +
                     w_tardiness * fitness_tardiness +
                     w_utilization * fitness_utilization +
                     w_priority * fitness_priority)
    
    return total_fitness

def selection_tournament(population: List[List[schemas.Operation]], fitness_scores: List[float], num_parents: int, tournament_size: int = 3) -> List[List[schemas.Operation]]:
    """Seleção por torneio."""
    selected_parents = []
    if not population or not fitness_scores: return []

    for _ in range(num_parents):
        # Ensure tournament_size is not greater than population size
        current_tournament_size = min(tournament_size, len(population))
        if current_tournament_size == 0 : continue # Should not happen if population is not empty

        tournament_indices = random.sample(range(len(population)), current_tournament_size)
        tournament_fitnesses = [(fitness_scores[i], population[i]) for i in tournament_indices]
        winner = max(tournament_fitnesses, key=lambda item: item[0])[1]
        selected_parents.append(winner)
    return selected_parents

def ordered_crossover(parent1: List[schemas.Operation], parent2: List[schemas.Operation]) -> Tuple[List[schemas.Operation], List[schemas.Operation]]:
    """Ordered Crossover (OX1) para permutações."""
    if not parent1 or not parent2 or len(parent1) != len(parent2) or len(parent1) == 0: # Added len(parent1) == 0 check
        return parent1[:], parent2[:] # Return copies

    size = len(parent1)
    child1, child2 = [-1]*size, [-1]*size 

    start, end = sorted(random.sample(range(size), 2))

    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    parent1_op_ids = [f"{op.cod_os_completo}_{op.seq}" for op in parent1]
    parent2_op_ids = [f"{op.cod_os_completo}_{op.seq}" for op in parent2]
    child1_segment_op_ids = {f"{op.cod_os_completo}_{op.seq}" for op in child1[start:end+1] if isinstance(op, schemas.Operation)} #check if op is an Operation
    child2_segment_op_ids = {f"{op.cod_os_completo}_{op.seq}" for op in child2[start:end+1] if isinstance(op, schemas.Operation)} #check if op is an Operation
    
    current_pos_child1 = (end + 1) % size
    for op_idx in range(size):
        parent2_element_idx = (end + 1 + op_idx) % size 
        op_p2 = parent2[parent2_element_idx]
        op_p2_id = parent2_op_ids[parent2_element_idx]

        if op_p2_id not in child1_segment_op_ids:
            while child1[current_pos_child1] != -1: 
                current_pos_child1 = (current_pos_child1 + 1) % size
            child1[current_pos_child1] = op_p2
            # current_pos_child1 = (current_pos_child1 + 1) % size # This line was causing issues by skipping positions
            
    current_pos_child2 = (end + 1) % size
    for op_idx in range(size):
        parent1_element_idx = (end + 1 + op_idx) % size
        op_p1 = parent1[parent1_element_idx]
        op_p1_id = parent1_op_ids[parent1_element_idx]

        if op_p1_id not in child2_segment_op_ids:
            while child2[current_pos_child2] != -1:
                current_pos_child2 = (current_pos_child2 + 1) % size
            child2[current_pos_child2] = op_p1
            # current_pos_child2 = (current_pos_child2 + 1) % size # This line was causing issues

    # Final check to fill any remaining -1 (should ideally not happen with correct logic)
    # This part is tricky, as simply filling with parent might break permutation property
    # For now, if -1 exists, it indicates a flaw in filling logic that needs deeper fix
    # For robustness, we might fallback to returning parents if children are malformed.
    if any(c == -1 for c in child1) or any(c == -1 for c in child2):
        # print("Alerta: Crossover falhou em preencher todos os genes. Retornando pais.")
        # Fallback to prevent returning incomplete children
        # Re-create children ensuring all elements from parents are present if -1 found
        
        # Fallback for child1
        if any(c == -1 for c in child1):
            filled_child1 = child1[start:end+1]
            child1_segment_op_ids_final = {f"{op.cod_os_completo}_{op.seq}" for op in filled_child1 if isinstance(op, schemas.Operation)}
            
            idx_parent2 = 0
            for i in range(size):
                if child1[i] == -1:
                    while parent2_op_ids[idx_parent2] in child1_segment_op_ids_final:
                        idx_parent2 = (idx_parent2 + 1) % size
                    child1[i] = parent2[idx_parent2]
                    child1_segment_op_ids_final.add(parent2_op_ids[idx_parent2]) # Add to segment to avoid reuse
                    idx_parent2 = (idx_parent2 + 1) % size


        # Fallback for child2
        if any(c == -1 for c in child2):
            filled_child2 = child2[start:end+1]
            child2_segment_op_ids_final = {f"{op.cod_os_completo}_{op.seq}" for op in filled_child2 if isinstance(op, schemas.Operation)}

            idx_parent1 = 0
            for i in range(size):
                if child2[i] == -1:
                    while parent1_op_ids[idx_parent1] in child2_segment_op_ids_final:
                        idx_parent1 = (idx_parent1 + 1) % size
                    child2[i] = parent1[idx_parent1]
                    child2_segment_op_ids_final.add(parent1_op_ids[idx_parent1])
                    idx_parent1 = (idx_parent1 + 1) % size
        
        # One last check, if still -1, then critical error, return parents
        if any(c == -1 for c in child1) or any(c == -1 for c in child2):
             return parent1[:], parent2[:]


    return child1, child2


def swap_mutation(individual: List[schemas.Operation]) -> List[schemas.Operation]:
    """Mutação por troca de duas posições."""
    if not individual or len(individual) < 2:
        return individual
        
    mutated_individual = individual[:]
    idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
    mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
    return mutated_individual
