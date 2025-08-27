#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Promedio móvil centrado con ventana de 3
std::vector<double> promedioMovil(const std::vector<int>& senal) {
    std::vector<double> resultado(senal.size(), 0.0);
    for (size_t i = 1; i < senal.size() - 1; ++i) {
        resultado[i] = static_cast<double>(senal[i - 1] + senal[i] + senal[i + 1]) / 3.0;
    }
    resultado[0] = senal[0]; // Mantener primer valor original
    resultado.back() = senal.back(); // Mantener último valor original
    return resultado;
}

// Filtro de mediana con ventana de 3
std::vector<int> filtroMediana(const std::vector<int>& senal) {
    std::vector<int> resultado(senal.size(), 0);
    for (size_t i = 1; i < senal.size() - 1; ++i) {
        std::vector<int> ventana = {senal[i - 1], senal[i], senal[i + 1]};
        std::sort(ventana.begin(), ventana.end());
        resultado[i] = ventana[1];
    }
    resultado[0] = senal[0];
    resultado.back() = senal.back();
    return resultado;
}

// Suavizado exponencial (EMA)
std::vector<double> calcularEMA(const std::vector<int>& senal, double alpha) {
    std::vector<double> ema;
    ema.reserve(senal.size()); // Mejora eficiencia
    ema.push_back(static_cast<double>(senal[0]));
    for (size_t t = 1; t < senal.size(); ++t) {
        double s_t = alpha * senal[t] + (1 - alpha) * ema.back();
        ema.push_back(s_t);
    }
    return ema;
}

int main() {
    constexpr int N = 120;         // Número de datos
    constexpr int MIN_VAL = 1;     // Valor mínimo
    constexpr int MAX_VAL = 100;   // Valor máximo
    constexpr double alpha = 0.3;

    // Inicializar generador aleatorio
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Generar señal aleatoria
    std::vector<int> senal(N);
    for (int& val : senal) {
        val = MIN_VAL + std::rand() % (MAX_VAL - MIN_VAL + 1);
    }

    const auto promedio = promedioMovil(senal);
    const auto mediana = filtroMediana(senal);
    const auto ema = calcularEMA(senal, alpha);

    std::cout << "Index\tOriginal\tPromedioMovil\tMediana\t\tEMA\n";
    for (size_t i = 0; i < senal.size(); ++i) {
        std::cout << i << "\t" << senal[i] << "\t\t"
                  << std::fixed << std::setprecision(2) << promedio[i] << "\t\t"
                  << mediana[i] << "\t\t" << ema[i] << "\n";
    }

    return 0;
}
