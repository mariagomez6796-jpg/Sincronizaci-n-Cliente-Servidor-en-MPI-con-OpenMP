#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <ctime>
#include <cstring>

using namespace std;

// ============================================================
// CONFIGURACION
// ============================================================

const char* NOMBRE_EQUIPO = "EQUIPO 3";

const double SALDO_INICIAL = 10000.0;

const int TAG_TRANSACCION = 100;
const int TAG_RESPUESTA   = 200;
const int TAG_FIN         = 300;

// Tipos de operacion
const int OP_DEPOSITO = 1;
const int OP_RETIRO   = 2;
const int OP_CONSULTA = 3;

// ============================================================
// ESTRUCTURAS
// ============================================================

struct Transaccion
{
    int id_cliente;
    int id_hilo_openmp;
    int numero_operacion;
    int tipo_operacion;
    double monto;
};

struct RespuestaServidor
{
    int aprobada;
    double saldo_resultante;
    char mensaje[100];
};

// ============================================================
// FUNCIONES AUXILIARES
// ============================================================

string nombreOperacion(int tipo)
{
    switch (tipo)
    {
        case OP_DEPOSITO:
            return "Deposito";

        case OP_RETIRO:
            return "Retiro";

        case OP_CONSULTA:
            return "Consulta";

        default:
            return "Desconocida";
    }
}

// ============================================================
// CREAR ARCHIVO LOG
// ============================================================

string crearNombreLog(const string& hostname, int rank)
{
    return "log_equipo_" +
           hostname +
           "_nodo_" +
           to_string(rank) +
           ".txt";
}

// ============================================================
// ENCABEZADO LOG
// ============================================================

void escribirEncabezadoLog(
    ofstream& log,
    int rank,
    int size,
    const string& hostname)
{
    log << "\n";
    log << "====================================================\n";
    log << " BANCO CLIENTE-SERVIDOR MPI + OPENMP\n";
    log << "====================================================\n";

    log << "Equipo: " << NOMBRE_EQUIPO << "\n";
    log << "Host: " << hostname << "\n";
    log << "Nodo MPI: " << rank << "\n";
    log << "Procesos MPI: " << size << "\n";
    log << "Hilos OpenMP disponibles: "
        << omp_get_max_threads() << "\n";

    if (rank == 0)
        log << "Rol: SERVIDOR\n";
    else
        log << "Rol: CLIENTE\n";

    log << "====================================================\n";
}

// ============================================================
// GENERAR TRANSACCIONES CON OPENMP
// ============================================================

void generarTransacciones(
    Transaccion* transacciones,
    int cantidad,
    int rank)
{
    unsigned int semillaBase =
        static_cast<unsigned int>(time(nullptr)) +
        rank * 10000;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        mt19937 generador(
            semillaBase +
            tid * 1000
        );

        uniform_int_distribution<int> tipoOperacion(1, 3);

        uniform_real_distribution<double> montoOperacion(
            100.0,
            2000.0
        );

        #pragma omp for
        for (int i = 0; i < cantidad; i++)
        {
            int tipo = tipoOperacion(generador);

            transacciones[i].id_cliente = rank;
            transacciones[i].id_hilo_openmp = tid;
            transacciones[i].numero_operacion = i + 1;
            transacciones[i].tipo_operacion = tipo;

            if (tipo == OP_CONSULTA)
            {
                transacciones[i].monto = 0.0;
            }
            else
            {
                transacciones[i].monto =
                    montoOperacion(generador);
            }
        }
    }
}

// ============================================================
// SERVIDOR
// ============================================================

void ejecutarServidor(
    int cantidadClientes,
    double& saldoCuenta,
    ofstream& log,
    bool mostrarDetalle,
    long long& totalProcesadas)
{
    int clientesFinalizados = 0;

    totalProcesadas = 0;

    while (clientesFinalizados < cantidadClientes)
    {
        MPI_Status status;

        MPI_Probe(
            MPI_ANY_SOURCE,
            MPI_ANY_TAG,
            MPI_COMM_WORLD,
            &status
        );

        // ====================================================
        // FIN DEL CLIENTE
        // ====================================================

        if (status.MPI_TAG == TAG_FIN)
        {
            MPI_Recv(
                nullptr,
                0,
                MPI_BYTE,
                status.MPI_SOURCE,
                TAG_FIN,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            clientesFinalizados++;

            log << "[SERVIDOR] Cliente MPI "
                << status.MPI_SOURCE
                << " finalizo sus operaciones.\n";

            continue;
        }

        // ====================================================
        // RECIBIR TRANSACCION
        // ====================================================

        if (status.MPI_TAG == TAG_TRANSACCION)
        {
            Transaccion transaccion;

            MPI_Recv(
                &transaccion,
                sizeof(Transaccion),
                MPI_BYTE,
                status.MPI_SOURCE,
                TAG_TRANSACCION,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            RespuestaServidor respuesta;

            respuesta.aprobada = 1;

            // ================================================
            // DEPOSITO
            // ================================================

            if (transaccion.tipo_operacion == OP_DEPOSITO)
            {
                saldoCuenta += transaccion.monto;

                strcpy_s(
                    respuesta.mensaje,
                    sizeof(respuesta.mensaje),
                    "Deposito aprobado"
                );
            }

            // ================================================
            // RETIRO
            // ================================================

            else if (
                transaccion.tipo_operacion == OP_RETIRO)
            {
                if (saldoCuenta >= transaccion.monto)
                {
                    saldoCuenta -= transaccion.monto;

                    strcpy_s(
                        respuesta.mensaje,
                        sizeof(respuesta.mensaje),
                        "Retiro aprobado"
                    );
                }
                else
                {
                    respuesta.aprobada = 0;

                    strcpy_s(
                        respuesta.mensaje,
                        sizeof(respuesta.mensaje),
                        "Retiro rechazado: fondos insuficientes"
                    );
                }
            }

            // ================================================
            // CONSULTA
            // ================================================

            else if (
                transaccion.tipo_operacion == OP_CONSULTA)
            {
                strcpy_s(
                    respuesta.mensaje,
                    sizeof(respuesta.mensaje),
                    "Consulta realizada"
                );
            }

            respuesta.saldo_resultante = saldoCuenta;

            totalProcesadas++;

            // ================================================
            // ENVIAR RESPUESTA
            // ================================================

            MPI_Send(
                &respuesta,
                sizeof(RespuestaServidor),
                MPI_BYTE,
                status.MPI_SOURCE,
                TAG_RESPUESTA,
                MPI_COMM_WORLD
            );

            // ================================================
            // LOG SERVIDOR
            // ================================================

            log << fixed << setprecision(2);

            log
                << "[Equipo: " << NOMBRE_EQUIPO << "] "
                << "[Nodo MPI: 0] "
                << "[Cliente: "
                << transaccion.id_cliente << "] "
                << "[Hilo OpenMP origen: "
                << transaccion.id_hilo_openmp << "] "
                << "[Operacion #: "
                << transaccion.numero_operacion << "] "
                << "[Op: "
                << nombreOperacion(
                    transaccion.tipo_operacion)
                << "] "
                << "[Monto: $"
                << transaccion.monto << "] "
                << "[Estado: "
                << (respuesta.aprobada
                    ? "Aprobado"
                    : "Rechazado")
                << "] "
                << "[Saldo: $"
                << respuesta.saldo_resultante
                << "]\n";

            // ================================================
            // CONSOLA PRIMERA EJECUCION
            // ================================================

            if (mostrarDetalle)
            {
                cout << fixed << setprecision(2);

                cout
                    << "[Servidor MPI 0]"
                    << " Cliente: "
                    << transaccion.id_cliente

                    << " | Hilo: "
                    << transaccion.id_hilo_openmp

                    << " | Op: "
                    << nombreOperacion(
                        transaccion.tipo_operacion)

                    << " | Monto: $"
                    << transaccion.monto

                    << " | Estado: "
                    << (respuesta.aprobada
                        ? "Aprobado"
                        : "Rechazado")

                    << " | Saldo: $"
                    << respuesta.saldo_resultante

                    << endl;
            }
        }
    }
}

// ============================================================
// CLIENTE
// ============================================================

void ejecutarCliente(
    int rank,
    int cantidadOperaciones,
    ofstream& log,
    bool mostrarDetalle)
{
    // ========================================================
    // MEMORIA DINAMICA
    // ========================================================

    Transaccion* historial =
        new Transaccion[cantidadOperaciones];

    // ========================================================
    // GENERAR OPERACIONES CON OPENMP
    // ========================================================

    generarTransacciones(
        historial,
        cantidadOperaciones,
        rank
    );

    // ========================================================
    // ENVIAR OPERACIONES
    // ========================================================

    for (int i = 0;
         i < cantidadOperaciones;
         i++)
    {
        MPI_Send(
            &historial[i],
            sizeof(Transaccion),
            MPI_BYTE,
            0,
            TAG_TRANSACCION,
            MPI_COMM_WORLD
        );

        RespuestaServidor respuesta;

        MPI_Recv(
            &respuesta,
            sizeof(RespuestaServidor),
            MPI_BYTE,
            0,
            TAG_RESPUESTA,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        // ====================================================
        // LOG CLIENTE
        // ====================================================

        log << fixed << setprecision(2);

        log
            << "[Equipo: " << NOMBRE_EQUIPO << "] "
            << "[Nodo MPI: " << rank << "] "
            << "[Hilo OpenMP: "
            << historial[i].id_hilo_openmp << "] "
            << "[Operacion #: "
            << historial[i].numero_operacion << "] "
            << "[Op: "
            << nombreOperacion(
                historial[i].tipo_operacion)
            << "] "
            << "[Monto: $"
            << historial[i].monto << "] "
            << "[Estado: "
            << (respuesta.aprobada
                ? "Aprobado"
                : "Rechazado")
            << "] "
            << "[Saldo: $"
            << respuesta.saldo_resultante
            << "]\n";

        // ====================================================
        // CONSOLA
        // ====================================================

        if (mostrarDetalle)
        {
            cout << fixed << setprecision(2);

            cout
                << "[Equipo: "
                << NOMBRE_EQUIPO << "] "

                << "[Nodo MPI: "
                << rank << "] "

                << "[Hilo OpenMP: "
                << historial[i].id_hilo_openmp
                << "] "

                << "[Operacion: "
                << nombreOperacion(
                    historial[i].tipo_operacion)
                << "] "

                << "[Monto: $"
                << historial[i].monto
                << "] "

                << "[Estado: "
                << (respuesta.aprobada
                    ? "Aprobado"
                    : "Rechazado")
                << "] "

                << "[Saldo: $"
                << respuesta.saldo_resultante
                << "]"

                << endl;
        }
    }

    // ========================================================
    // AVISAR FIN
    // ========================================================

    MPI_Send(
        nullptr,
        0,
        MPI_BYTE,
        0,
        TAG_FIN,
        MPI_COMM_WORLD
    );

    // ========================================================
    // LIBERAR MEMORIA
    // ========================================================

    delete[] historial;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char* argv[])
{
    // ========================================================
    // INICIAR MPI
    // ========================================================

    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &size
    );

    // ========================================================
    // OBTENER NOMBRE DEL EQUIPO
    // ========================================================

    char processorName[MPI_MAX_PROCESSOR_NAME];

    int nameLength;

    MPI_Get_processor_name(
        processorName,
        &nameLength
    );

    string hostname(processorName);

    // ========================================================
    // CREAR LOG LOCAL
    // ========================================================

    string nombreLog =
        crearNombreLog(
            hostname,
            rank
        );

    ofstream log(
        nombreLog,
        ios::app
    );

    escribirEncabezadoLog(
        log,
        rank,
        size,
        hostname
    );

    // ========================================================
    // VALIDAR NUMERO DE PROCESOS
    // ========================================================

    if (size < 2)
    {
        if (rank == 0)
        {
            cout
                << "ERROR: Se necesita minimo "
                << "1 servidor y 1 cliente."
                << endl;
        }

        log.close();

        MPI_Finalize();

        return 0;
    }

    // ========================================================
    // SALDO DEL SERVIDOR
    // ========================================================

    double saldoCuenta =
        SALDO_INICIAL;

    int opcion = 0;

    // ========================================================
    // MENU
    // ========================================================

    do
    {
        if (rank == 0)
        {
            cout << "\n";
            cout << "========================================\n";
            cout << " BANCO CLIENTE-SERVIDOR MPI + OPENMP\n";
            cout << "========================================\n";
            cout << "Equipo: " << NOMBRE_EQUIPO << "\n";
            cout << "Servidor: Nodo MPI 0\n";
            cout << "Clientes: " << size - 1 << "\n";

            cout << fixed << setprecision(2);

            cout
                << "Saldo actual: $"
                << saldoCuenta
                << "\n";

            cout << "========================================\n";
            cout << "1. Simulacion basica (5 operaciones)\n";
            cout << "2. Simulacion masiva (100000 operaciones)\n";
            cout << "3. Consultar saldo final\n";
            cout << "4. Salir\n";
            cout << "========================================\n";
            cout << "Opcion: ";

            cin >> opcion;
        }

        // ====================================================
        // DISTRIBUIR OPCION
        // ====================================================

        MPI_Bcast(
            &opcion,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        // ====================================================
        // OPCION 1
        // ====================================================

        if (opcion == 1)
        {
            const int operaciones = 5;

            MPI_Barrier(
                MPI_COMM_WORLD
            );

            double inicio =
                MPI_Wtime();

            long long totalProcesadas = 0;

            if (rank == 0)
            {
                cout << "\n";
                cout << "========================================\n";
                cout << " PRIMERA EJECUCION\n";
                cout << "========================================\n";
                cout << "5 operaciones por cliente\n";
                cout << "Clientes: "
                     << size - 1
                     << "\n";
                cout << "========================================\n\n";

                ejecutarServidor(
                    size - 1,
                    saldoCuenta,
                    log,
                    true,
                    totalProcesadas
                );
            }
            else
            {
                ejecutarCliente(
                    rank,
                    operaciones,
                    log,
                    true
                );
            }

            MPI_Barrier(
                MPI_COMM_WORLD
            );

            double fin =
                MPI_Wtime();

            if (rank == 0)
            {
                cout << "\n";
                cout << "========================================\n";
                cout << " RESULTADOS PRIMERA EJECUCION\n";
                cout << "========================================\n";

                cout
                    << "Transacciones procesadas: "
                    << totalProcesadas
                    << "\n";

                cout << fixed << setprecision(2);

                cout
                    << "Saldo final: $"
                    << saldoCuenta
                    << "\n";

                cout << setprecision(6);

                cout
                    << "Tiempo total: "
                    << fin - inicio
                    << " segundos\n";

                cout << "========================================\n";

                log
                    << "\nTiempo primera ejecucion: "
                    << fin - inicio
                    << " segundos\n";

                log
                    << "Saldo final: $"
                    << saldoCuenta
                    << "\n";
            }
        }

        // ====================================================
        // OPCION 2
        // ====================================================

        else if (opcion == 2)
        {
            const int operaciones = 100000;

            MPI_Barrier(
                MPI_COMM_WORLD
            );

            double inicio =
                MPI_Wtime();

            long long totalProcesadas = 0;

            if (rank == 0)
            {
                cout << "\n";
                cout << "Iniciando simulacion masiva...\n";

                ejecutarServidor(
                    size - 1,
                    saldoCuenta,
                    log,
                    false,
                    totalProcesadas
                );
            }
            else
            {
                ejecutarCliente(
                    rank,
                    operaciones,
                    log,
                    false
                );
            }

            MPI_Barrier(
                MPI_COMM_WORLD
            );

            double fin =
                MPI_Wtime();

            if (rank == 0)
            {
                cout << "\n";
                cout << "========================================\n";
                cout << " RESULTADOS SIMULACION MASIVA\n";
                cout << "========================================\n";

                cout
                    << "Clientes: "
                    << size - 1
                    << "\n";

                cout
                    << "Operaciones por cliente: "
                    << operaciones
                    << "\n";

                cout
                    << "Total procesadas: "
                    << totalProcesadas
                    << "\n";

                cout << fixed << setprecision(2);

                cout
                    << "Saldo final: $"
                    << saldoCuenta
                    << "\n";

                cout << setprecision(6);

                cout
                    << "Tiempo total: "
                    << fin - inicio
                    << " segundos\n";

                cout << "========================================\n";

                log
                    << "\nSIMULACION MASIVA\n";

                log
                    << "Total procesadas: "
                    << totalProcesadas
                    << "\n";

                log
                    << "Tiempo: "
                    << fin - inicio
                    << " segundos\n";

                log
                    << "Saldo final: $"
                    << saldoCuenta
                    << "\n";
            }
        }

        // ====================================================
        // OPCION 3
        // ====================================================

        else if (opcion == 3)
        {
            if (rank == 0)
            {
                cout << fixed << setprecision(2);

                cout << "\n";
                cout << "========================================\n";
                cout << " SALDO ACTUAL DEL SERVIDOR\n";
                cout << "========================================\n";
                cout << "Saldo: $"
                     << saldoCuenta
                     << "\n";
                cout << "========================================\n";
            }
        }

        // ====================================================
        // OPCION INVALIDA
        // ====================================================

        else if (opcion != 4)
        {
            if (rank == 0)
            {
                cout
                    << "Opcion invalida."
                    << endl;
            }
        }

    } while (opcion != 4);

    // ========================================================
    // FINAL
    // ========================================================

    log
        << "\nNodo MPI "
        << rank
        << " finalizado.\n";

    log.close();

    if (rank == 0)
    {
        cout << "\nPrograma finalizado.\n";
    }

    MPI_Finalize();

    return 0;
}