from datetime import datetime, timedelta, date, time

# Constantes
HORAS_TRABALHO_POR_DIA = 18.5
INICIO_TURNO = time(7, 30) # Exemplo: 07:30
FIM_TURNO = time(2, 0) # Exemplo: 02:00 do dia seguinte (considerando turno que vira a noite)
# Se o turno é contínuo dentro de um dia, ex: 08:00 às 18:00, ajustar FIM_TURNO e lógica.
# Para 18.5h, poderia ser das 07:30 às 02:00 do dia seguinte (com intervalo).
# Simplificação: assumindo 18.5h contínuas a partir do início do dia útil.

def eh_dia_util(data_atual: date) -> bool:
    """Verifica se é um dia útil (Segunda a Sexta)."""
    return data_atual.weekday() < 5  # Segunda é 0, Domingo é 6

def proximo_dia_util(data_atual: date) -> date:
    """Retorna o próximo dia útil."""
    proximo_dia = data_atual + timedelta(days=1)
    while not eh_dia_util(proximo_dia):
        proximo_dia += timedelta(days=1)
    return proximo_dia

def adicionar_horas_uteis(data_inicio: datetime, horas_a_adicionar: float) -> datetime:
    """
    Adiciona horas a uma data/hora, considerando apenas o turno de trabalho em dias úteis.
    Esta é uma função simplificada e pode precisar de ajustes para casos complexos de turno.
    """
    data_fim = data_inicio
    horas_restantes = horas_a_adicionar

    # Se a data de início não tem hora ou é antes do início do turno no primeiro dia, ajusta para o início do turno.
    if data_fim.time() == time(0,0) or (data_fim.date() == data_inicio.date() and data_fim.time() < INICIO_TURNO):
        current_date_to_use = data_fim.date()
        if not eh_dia_util(current_date_to_use):
            current_date_to_use = proximo_dia_util(current_date_to_use)
        data_fim = datetime.combine(current_date_to_use, INICIO_TURNO)
    
    while horas_restantes > 0:
        # Garante que data_fim está em um dia útil.
        if not eh_dia_util(data_fim.date()):
            data_fim = datetime.combine(proximo_dia_util(data_fim.date()), INICIO_TURNO)
            continue

        # Se data_fim (hora atual) é antes do INICIO_TURNO no dia atual (e não é meia-noite, que é tratado acima),
        # ajusta data_fim para o INICIO_TURNO do dia atual.
        if data_fim.time() < INICIO_TURNO and data_fim.time() != time(0,0):
             data_fim = datetime.combine(data_fim.date(), INICIO_TURNO)

        # Determina o início e o fim reais do período de turno atual
        current_shift_start_dt = datetime.combine(data_fim.date(), INICIO_TURNO)
        current_shift_end_dt = datetime.combine(data_fim.date(), FIM_TURNO)

        if FIM_TURNO < INICIO_TURNO: # Turno da noite (ex: 07:30 - 02:00 do dia seguinte)
            if data_fim.time() < FIM_TURNO: # Estamos na parte da manhã do turno (ex: 01:00)
                # O turno começou no dia anterior
                current_shift_start_dt = datetime.combine(data_fim.date() - timedelta(days=1), INICIO_TURNO)
            else: # Estamos na parte da tarde/noite do turno (ex: 08:00)
                # O turno termina no dia seguinte
                current_shift_end_dt = datetime.combine(data_fim.date() + timedelta(days=1), FIM_TURNO)
        
        # Se data_fim está fora da janela do turno atual (depois que o turno termina),
        # avança para o próximo turno.
        if data_fim >= current_shift_end_dt:
            # Determina a data base para encontrar o próximo dia útil
            base_date_for_next_shift = data_fim.date()
            # Se o turno terminou e era um turno que vira a noite (FIM_TURNO < INICIO_TURNO),
            # o dia atual já é o dia em que o turno terminou.
            # Se o turno terminou e era no mesmo dia, precisamos do próximo dia útil a partir do dia atual.
            if FIM_TURNO >= INICIO_TURNO: # Turno no mesmo dia
                 base_date_for_next_shift = proximo_dia_util(data_fim.date())
            else: # Turno vira noite
                # Se data_fim.time() < INICIO_TURNO, significa que já estamos no dia seguinte ao início do turno noturno
                # e o current_shift_end_dt já foi calculado corretamente para este dia.
                # Se data_fim.time() >= INICIO_TURNO, o turno começou neste dia e termina no próximo.
                # A data_fim já está correta para buscar o próximo dia útil se necessário.
                pass # base_date_for_next_shift já é data_fim.date()

            next_processing_day = base_date_for_next_shift
            if not eh_dia_util(next_processing_day):
                next_processing_day = proximo_dia_util(next_processing_day)
            
            data_fim = datetime.combine(next_processing_day, INICIO_TURNO)
            continue

        # Calcula horas disponíveis no período de turno válido atual
        # data_fim agora está garantido estar dentro de um slot de turno válido ou no seu início.
        horas_disponiveis_no_turno_atual = (current_shift_end_dt - data_fim).total_seconds() / 3600
        horas_a_adicionar_neste_ciclo = min(horas_restantes, horas_disponiveis_no_turno_atual)
        horas_a_adicionar_neste_ciclo = max(0, horas_a_adicionar_neste_ciclo) # Garante não ser negativo

        data_fim += timedelta(hours=horas_a_adicionar_neste_ciclo)
        horas_restantes -= horas_a_adicionar_neste_ciclo

        # Se ainda há horas restantes, significa que o turno atual terminou.
        # Prepara para o próximo ciclo, movendo para o início do próximo turno.
        if horas_restantes > 0:
            # Avança para o início do próximo turno.
            # A data base para o próximo turno é o dia em que o turno atual terminou.
            date_for_next_shift_start = current_shift_end_dt.date()
            if not eh_dia_util(date_for_next_shift_start):
                 date_for_next_shift_start = proximo_dia_util(date_for_next_shift_start)
            elif current_shift_end_dt.time() >= FIM_TURNO and FIM_TURNO < INICIO_TURNO : # Fim do turno noturno
                 pass # a data já é o dia seguinte ao início do turno. Se não for dia útil, será corrigido.
            elif current_shift_end_dt.time() >= FIM_TURNO : # Fim do turno diurno ou já no dia seguinte ao noturno
                 date_for_next_shift_start = proximo_dia_util(current_shift_end_dt.date())


            data_fim = datetime.combine(date_for_next_shift_start, INICIO_TURNO)
            # Garante que o próximo início seja dia útil
            if not eh_dia_util(data_fim.date()):
                data_fim = datetime.combine(proximo_dia_util(data_fim.date()), INICIO_TURNO)

    return data_fim

def adicionar_dias_uteis(data_inicio: date, dias_a_adicionar: int) -> date:
    """Adiciona dias úteis a uma data."""
    data_fim = data_inicio
    dias_adicionados = 0
    while dias_adicionados < dias_a_adicionar:
        data_fim += timedelta(days=1)
        if eh_dia_util(data_fim):
            dias_adicionados += 1
    return data_fim

def calculate_total_working_hours(period_start_dt: datetime, period_end_dt: datetime) -> float:
    if period_start_dt >= period_end_dt:
        return 0.0

    total_hours = 0.0
    current_dt = period_start_dt

    # Adjust start to be within a working shift.
    # adicionar_horas_uteis(current_dt, 0) finds the start of the current or next available slot.
    current_dt = adicionar_horas_uteis(current_dt, 0)

    while current_dt < period_end_dt:
        # This loop processes one valid working shift (or part of it) at a time.
        if not eh_dia_util(current_dt.date()): # Should not happen if adicionar_horas_uteis works
            current_dt = datetime.combine(proximo_dia_util(current_dt.date()), INICIO_TURNO)
            current_dt = adicionar_horas_uteis(current_dt, 0) # Re-align current_dt
            if current_dt >= period_end_dt: break # Past the period end after adjustment
            continue

        # Define the conceptual shift boundaries for current_dt's date
        # This is the shift that current_dt is currently part of, or the next one if current_dt was adjusted.
        
        # Determine the actual start and end of the specific shift instance that current_dt falls into.
        # current_dt is already guaranteed to be at the beginning of a valid work slot by adicionar_horas_uteis.
        shift_instance_start = current_dt
        
        # Calculate the end of this specific shift instance.
        # If FIM_TURNO < INICIO_TURNO (overnight):
        #   If current_dt.time() is in the "first part" (e.g., 7:30 to 23:59), shift ends next day at FIM_TURNO.
        #   If current_dt.time() is in the "second part" (e.g., 00:00 to 02:00), shift ends same day at FIM_TURNO.
        # Else (same-day shift):
        #   Shift ends same day at FIM_TURNO.
        
        shift_instance_end_date = shift_instance_start.date()
        if FIM_TURNO < INICIO_TURNO and shift_instance_start.time() >= INICIO_TURNO:
            shift_instance_end_date += timedelta(days=1)
        # For same-day shifts, or the "morning part" of an overnight shift, date is correct.
        
        shift_instance_end = datetime.combine(shift_instance_end_date, FIM_TURNO)

        # Determine how much of this shift is within the overall [period_start_dt, period_end_dt]
        effective_start_this_segment = shift_instance_start # Already max(current_dt, shift_instance_start)
        effective_end_this_segment = min(period_end_dt, shift_instance_end)

        if effective_start_this_segment < effective_end_this_segment:
            total_hours += (effective_end_this_segment - effective_start_this_segment).total_seconds() / 3600.0
        
        # Move current_dt to the start of the next shift period for the next iteration.
        # This will be after shift_instance_end.
        # Use adicionar_horas_uteis with 0 to find the very beginning of the next valid period.
        current_dt = adicionar_horas_uteis(shift_instance_end, 0)
        
        # If the next shift starts at or after period_end_dt, we are done.
        if current_dt >= period_end_dt:
            break
            
    return round(total_hours, 2)
