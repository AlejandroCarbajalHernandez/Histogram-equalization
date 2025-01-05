#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <limits.h> //Limites de los tipos de datos
#include <sys/stat.h>  //Para crear el directorio
#include <sys/types.h> //Tipos de datos para estructuras

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb-master/stb_image.h"
#include "stb-master/stb_image_write.h"

//Prototipos de función
void calculate_histogram(const unsigned char *image, int width, int height, int channels, int *histogram);
void calculate_cdf(const int *histogram, int *cdf);
int calculate_cdf_min(const int *cdf);
void calculate_equalized_cdf(const int *cdf, int cdf_min, int total_pixels, int *equalized_cdf);
void apply_equalization(const unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, const int *equalized_cdf);
void write_csv(const char *filename, const int *original_histogram, const int *equalized_histogram);
void generate_histogram_image(const int *histogram, const char *filename);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <ruta_imagen_entrada>\n", argv[0]);
        return -1;
    }

    // Crear el directorio 'resultados' si no existe
    struct stat st = {0}; //stat verifica el directorio "resultados"
    if (stat("resultados", &st) == -1) {
        #ifdef _WIN32 //Compilador para Windows
            mkdir("resultados");
        #else
            mkdir("resultados", 0700); //Compilador para Linux/Mac
        // 0700 son los permisos iniciales del directorio
        #endif

    }

    //Variables para medir tiempos
    double overhead_start_time, overhead_end_time;
    double image_load_time = 0.0, image_save_time = 0.0, csv_generation_time = 0.0;
    overhead_start_time = omp_get_wtime();

    //Obtener el número de procesadores disponibles
    int num_processors = omp_get_num_procs();
    printf("Numero de procesadores: %d\n", num_processors);

    //Cargar la imagen
    double load_start_time = omp_get_wtime();
    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error: No se pudo cargar la imagen.\n");
        return -1;
    }
    image_load_time = omp_get_wtime() - load_start_time;
    printf("Imagen cargada: %s (Ancho: %d, Alto: %d, Canales: %d)\n", argv[1], width, height, channels);
    printf("Tamaño de la imagen: %d bytes\n", width * height * channels);

    //Verificar y ajustar canales si es necesario para JPG
    int adjusted_channels = channels;
    if (channels == 4) {
        adjusted_channels = 3; //JPG no soporta canal alfa
        printf("Nota: La imagen tiene 4 canales. El canal alfa se descartara al guardar en JPG.\n");
    }

    int total_pixels = width * height * adjusted_channels;
    //Inicialización de arreglos
    int histogram[256] = {0}, cdf[256] = {0}, equalized_cdf[256] = {0};
    //Reservar memoria suficiente, dinámicamente
    unsigned char *equalized_image = malloc(width * height * adjusted_channels);

    double start_time, sequential_time, parallel_time;

    //Obtener el nombre base del archivo sin extensión y sin la ruta
    char base_name[256];
    char *filename = strrchr(argv[1], '/'); //Búsqueda del carácter
    if (filename)
        filename++;
    else
        filename = argv[1];

    strcpy(base_name, filename);
    char *dot = strrchr(base_name, '.');
    if (dot) *dot = '\0';

    //Implementación secuencial
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    printf("\nImplementacion secuencial:\n");
    printf("  Numero maximo de hilos disponibles: %d\n", omp_get_max_threads());

    start_time = omp_get_wtime();

    //Calcular histograma de la imagen original
    calculate_histogram(image, width, height, adjusted_channels, histogram);

    //Calcular CDF y ecualización para la imagen completa
    calculate_cdf(histogram, cdf);
    int cdf_min = calculate_cdf_min(cdf);
    calculate_equalized_cdf(cdf, cdf_min, width * height * adjusted_channels, equalized_cdf);

    //Aplicar ecualización
    apply_equalization(image, equalized_image, width, height, adjusted_channels, equalized_cdf);

    sequential_time = omp_get_wtime() - start_time;

    //Guardar salidas secuenciales (fuera del tiempo medido)
    double save_start_time = omp_get_wtime();

    char output_image_name[512];
    snprintf(output_image_name, sizeof(output_image_name), "resultados/%s_eq_secuencial.jpg", base_name);
    stbi_write_jpg(output_image_name, width, height, adjusted_channels, equalized_image, 100);
    image_save_time += omp_get_wtime() - save_start_time;

    double csv_start_time = omp_get_wtime();
    char csv_file_name[512];
    snprintf(csv_file_name, sizeof(csv_file_name), "resultados/%s_histo_secuencial.csv", base_name);
    write_csv(csv_file_name, histogram, equalized_cdf);
    csv_generation_time += omp_get_wtime() - csv_start_time;

    //Generar imágenes de histogramas
    char histogram_image_name_original[512];
    snprintf(histogram_image_name_original, sizeof(histogram_image_name_original), "resultados/%s_histogram_original.jpg", base_name);
    generate_histogram_image(histogram, histogram_image_name_original);

    //Calcular histograma de la imagen ecualizada para generar su histograma
    int equalized_histogram[256] = {0};
    calculate_histogram(equalized_image, width, height, adjusted_channels, equalized_histogram);

    char histogram_image_name_equalized[512];
    snprintf(histogram_image_name_equalized, sizeof(histogram_image_name_equalized), "resultados/%s_histogram_eq_secuencial.jpg", base_name);
    generate_histogram_image(equalized_histogram, histogram_image_name_equalized);

    //Implementación paralela
    memset(histogram, 0, sizeof(histogram)); //Reiniciar histograma
    memset(cdf, 0, sizeof(cdf));
    memset(equalized_cdf, 0, sizeof(equalized_cdf));
    memset(equalized_histogram, 0, sizeof(equalized_histogram));

    omp_set_dynamic(0);
    int max_threads = num_processors; //Obtener el número de procesadores disponibles
    omp_set_num_threads(max_threads);
    printf("\nImplementacion paralela:\n");
    printf("  Numero maximo de hilos disponibles: %d\n", omp_get_max_threads());

    start_time = omp_get_wtime();

    //Calcular histograma de la imagen original
    calculate_histogram(image, width, height, adjusted_channels, histogram);

    //Calcular CDF y ecualización para la imagen completa
    calculate_cdf(histogram, cdf);
    cdf_min = calculate_cdf_min(cdf);
    calculate_equalized_cdf(cdf, cdf_min, width * height * adjusted_channels, equalized_cdf);

    //Aplicar ecualización
    apply_equalization(image, equalized_image, width, height, adjusted_channels, equalized_cdf);

    parallel_time = omp_get_wtime() - start_time;

    //Guardar salidas paralelas (fuera del tiempo medido)
    save_start_time = omp_get_wtime();
    snprintf(output_image_name, sizeof(output_image_name), "resultados/%s_eq_paralelo.jpg", base_name);
    stbi_write_jpg(output_image_name, width, height, adjusted_channels, equalized_image, 100);
    image_save_time += omp_get_wtime() - save_start_time;

    csv_start_time = omp_get_wtime();
    snprintf(csv_file_name, sizeof(csv_file_name), "resultados/%s_histo_paralelo.csv", base_name);
    write_csv(csv_file_name, histogram, equalized_cdf);
    csv_generation_time += omp_get_wtime() - csv_start_time;

    //Calcular histograma de la imagen ecualizada para generar su histograma
    calculate_histogram(equalized_image, width, height, adjusted_channels, equalized_histogram);

    snprintf(histogram_image_name_equalized, sizeof(histogram_image_name_equalized), "resultados/%s_histogram_eq_paralelo.jpg", base_name);
    generate_histogram_image(equalized_histogram, histogram_image_name_equalized);

    //Calcular tiempo de overhead
    overhead_end_time = omp_get_wtime();
    double overhead_time = overhead_end_time - overhead_start_time - image_load_time - image_save_time - csv_generation_time - sequential_time - parallel_time;

    //Imprimir métricas de rendimiento
    int num_threads = max_threads;
    double speedup = sequential_time / parallel_time;
    double efficiency = (speedup / num_threads) * 100;

    printf("\nMetricas de Rendimiento (Solo Algoritmo):\n");
    printf("  Tiempo Secuencial: %.6f segundos\n", sequential_time);
    printf("  Tiempo Paralelo: %.6f segundos\n", parallel_time);
    printf("  SpeedUp: %.2f\n", speedup);
    printf("  Eficiencia: %.2f%%\n", efficiency);

    printf("\nTiempos Adicionales:\n");
    printf("  Tiempo de Overhead: %.6f segundos\n", overhead_time);
    printf("  Tiempo de carga de imagen: %.6f segundos\n", image_load_time);
    printf("  Tiempo de generacion de imagen: %.6f segundos\n", image_save_time);
    printf("  Tiempo de generacion de archivos CSV: %.6f segundos\n", csv_generation_time);

    //Liberar memoria
    stbi_image_free(image);
    free(equalized_image);

    return 0;
}

//Calcula el histograma de la imagen completa
void calculate_histogram(const unsigned char *image, int width, int height, int channels, int *histogram) {
    int num_threads = omp_get_max_threads();
    //Declara puntero a puntero tipo int, estructura como matriz dinámica bidimensional
    int **local_hists = malloc(num_threads * sizeof(int *));
    for (int i = 0; i < num_threads; i++) {
        local_hists[i] = calloc(256, sizeof(int)); //Mem. dinámica
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int *local_hist = local_hists[thread_id];

        #pragma omp for nowait
        for (int i = 0; i < width * height * channels; i++) {
            local_hist[image[i]]++;
        }
    }

    //Combinar histogramas locales
    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < 256; j++) {
            histogram[j] += local_hists[i][j];
        }
        free(local_hists[i]);
    }
    free(local_hists);
}

void calculate_cdf(const int *histogram, int *cdf) {
    cdf[0] = histogram[0]; //Inicializa el primer valor del CDF con el primer valor del histograma.
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + histogram[i]; //Calcula la suma acumulada para cada nivel de intensidad.
    }
}

int calculate_cdf_min(const int *cdf) {
    int cdf_min = INT_MAX; //Inicializa cdf_min con el valor máximo posible para buscar el mínimo.

    #pragma omp parallel for reduction(min:cdf_min)
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0 && cdf[i] < cdf_min) {
            cdf_min = cdf[i]; //Encuentra el primer valor no nulo más bajo del CDF.
        }
    }
    return cdf_min; //Devuelve el valor mínimo del CDF.
}

void calculate_equalized_cdf(const int *cdf, int cdf_min, int total_pixels, int *equalized_cdf) {
    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
        equalized_cdf[i] = (int)round(((double)(cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255.0); //Reescala el CDF a un rango de 0 a 255.
        if (equalized_cdf[i] < 0)
            equalized_cdf[i] = 0; //Asegura que los valores negativos se conviertan a 0.
    }
}

void apply_equalization(const unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, const int *equalized_cdf) {
    #pragma omp parallel for
    for (int i = 0; i < width * height * channels; i++) {
        output_image[i] = (unsigned char)equalized_cdf[input_image[i]]; //Aplica la ecualización de intensidad a cada píxel de la imagen.
    }
}

void write_csv(const char *filename, const int *original_histogram, const int *equalized_cdf) {
    FILE *csv_file = fopen(filename, "w"); //Abre o crea un archivo CSV para escritura.
    if (!csv_file) {
        printf("Error: No se pudo crear el archivo CSV: %s\n", filename); //Verifica errores al abrir el archivo.
        return;
    }
    fprintf(csv_file, "Valor,Original,Ecualizado\n"); //Escribe la cabecera del CSV.
    for (int i = 0; i < 256; i++) {
        fprintf(csv_file, "%d,%d,%d\n", i, original_histogram[i], equalized_cdf[i]); //Escribe los datos del histograma y CDF ecualizado.
    }
    fclose(csv_file); //Cierra el archivo.
}

void generate_histogram_image(const int *histogram, const char *filename) {
    int width = 256; //Ancho de la imagen del histograma, un píxel por nivel de intensidad.
    int height = 200; //Altura de la imagen del histograma.
    unsigned char *image = calloc(width * height, sizeof(unsigned char)); //Reserva memoria para la imagen en escala de grises.

    int max_value = 0; //Inicializa el valor máximo del histograma para normalización.
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > max_value) {
            max_value = histogram[i]; //Encuentra el valor máximo del histograma.
        }
    }

    if (max_value == 0) {
        max_value = 1; //Evita divisiones por cero si el histograma está vacío.
    }

    for (int x = 0; x < width; x++) {
        int value = histogram[x];
        int bar_height = (int)(((double)value / max_value) * (height - 1)); //Calcula la altura de la barra normalizada.

        for (int y = 0; y <= bar_height; y++) {
            image[(height - y - 1) * width + x] = 255; //Dibuja la barra en blanco desde la parte inferior.
        }
    }

    unsigned char *rgb_image = malloc(width * height * 3); //Reserva memoria para convertir la imagen a formato RGB.
    for (int i = 0; i < width * height; i++) {
        rgb_image[i * 3] = image[i]; //Rellena el canal Rojo.
        rgb_image[i * 3 + 1] = image[i]; //Rellena el canal Verde.
        rgb_image[i * 3 + 2] = image[i]; //Rellena el canal Azul.
    }

    stbi_write_jpg(filename, width, height, 3, rgb_image, 100); //Guarda la imagen como un archivo JPG.

    free(image); //Libera la memoria reservada para la imagen en escala de grises.
    free(rgb_image); //Libera la memoria reservada para la imagen en formato RGB.
}
