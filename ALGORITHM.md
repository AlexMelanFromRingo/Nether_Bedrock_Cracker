# Алгоритм поиска сида по паттерну бедрока

## Обзор

Nether Bedrock Cracker использует паттерн генерации бедрока в Нижнем мире Minecraft для восстановления сида мира. Алгоритм основан на обратном инжиниринге Java LCG (Linear Congruential Generator), используемого Minecraft для генерации мира.

## Генерация бедрока в Minecraft

### Как Minecraft генерирует бедрок

```
World Seed
    │
    ▼
Random.setSeed(seed)
Random.nextLong()
    │
    ▼
Bedrock Seed (общий для floor и roof)
    │
    ├──► XOR с FLOOR_HASH (2042456806) ──► Floor Random
    │                                           │
    │                                           ▼
    │                                      nextLong()
    │                                           │
    │                                           ▼
    │                                      Floor Seed (48 бит)
    │
    └──► XOR с ROOF_HASH (343340730) ───► Roof Random
                                              │
                                              ▼
                                         nextLong()
                                              │
                                              ▼
                                         Roof Seed (48 бит)
```

### Генерация блока бедрока

Для каждой позиции `(x, y, z)`:

```java
// Вычисление хеша позиции
long posHash = (x * 3129871) ^ (z * 116129781L) ^ y;
posHash = posHash * posHash * 42317861L + posHash * 11L;
posHash = posHash >> 16;

// Инициализация Random
Random random = new Random(seed ^ posHash);
random.nextDouble();  // Пропуск первого значения

// Определение типа блока
double threshold = random.nextDouble();
double chance = getBedrockChance(y);  // Зависит от высоты

if (threshold < chance) {
    // Блок = BEDROCK
} else {
    // Блок = OTHER (netherrack, воздух и т.д.)
}
```

### Вероятности по высоте

**Потолок (y = 123-127):**
| Y   | Вероятность бедрока |
|-----|---------------------|
| 127 | 100%                |
| 126 | 80%                 |
| 125 | 60%                 |
| 124 | 40%                 |
| 123 | 20%                 |

**Пол (y = 0-4):**
| Y | Вероятность бедрока |
|---|---------------------|
| 0 | 100%                |
| 1 | 80%                 |
| 2 | 60%                 |
| 3 | 40%                 |
| 4 | 20%                 |

## Алгоритм поиска

### Этап 1: Подготовка CheckObjects

Для каждого блока создаётся `CheckObject`:

```rust
struct CheckObject {
    pos_hash: u64,    // Хеш позиции XOR с LCG multiplier
    condition: u64,   // Граница для проверки
    offset: u64,      // Смещение для нормализации
}
```

**Математика проверки:**

```
result = (upper_bits XOR pos_hash) * LCG_MULT + offset
result = result AND MASK48

Если result < condition:
    Блок НЕ соответствует ожидаемому типу → сид отклоняется
```

### Этап 2: Построение дерева фильтров (CPU)

Алгоритм строит дерево из 13 уровней (0-12 бит):

```
Уровень 12: Проверяем верхние 36 бит (2^36 итераций)
     │
     ├── Фильтр 1: Отбрасывает ~X% сидов
     ├── Фильтр 2: Отбрасывает ~Y% сидов
     └── ...
     │
     ▼
Уровень 11: Для каждого прошедшего сида проверяем 2 варианта
     │
     ├── upper_bits
     └── upper_bits + (1 << 11)
     │
     ▼
... продолжаем до уровня 0 ...
     │
     ▼
Уровень 0: CrossComparison
```

### Этап 3: CrossComparison

Когда сид проходит все фильтры первичной поверхности (floor или roof):

1. **Reverse NextLong** — находим возможные предыдущие состояния PRNG
2. **XOR с primary_hash** — получаем общий bedrock seed
3. **Проверка вторичной поверхности** — проверяем блоки на другой поверхности
4. **Reverse к world seed** — восстанавливаем оригинальный сид мира

```
Floor/Roof Seed (48 бит)
        │
        ▼
  reverse_next_long()
        │
        ▼
  ~4 кандидата
        │
        ▼
  XOR primary_hash
        │
        ▼
  Bedrock Seed
        │
        ├──► Проверка secondary surface
        │         │
        │         ▼
        │    Фильтрация
        │
        ▼
  reverse_next_long()
        │
        ▼
  Structure Seed
        │
        ▼
  reverse_next_long()
        │
        ▼
  World Seed
```

## GPU-ускорение

### Архитектура GPU Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU (OpenCL)                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Kernel: bedrock_filter                                   │  │
│  │                                                           │  │
│  │  for each work_item (1M параллельно):                    │  │
│  │      upper_bits = start + work_id << 12                  │  │
│  │                                                           │  │
│  │      for each check in checks:                           │  │
│  │          if check_single(upper_bits, check):             │  │
│  │              return  // Сид отклонён                     │  │
│  │                                                           │  │
│  │      // Сид прошёл все проверки                          │  │
│  │      candidates[atomic_inc(count)] = upper_bits          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    ~100-1000 кандидатов                        │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CPU                                     │
│                                                                 │
│  for candidate in candidates:                                   │
│      reverse_next_long(candidate)                              │
│      CrossComparison::run()                                    │
│      → World Seed                                              │
└─────────────────────────────────────────────────────────────────┘
```

### OpenCL Kernel

```opencl
#define JAVA_LCG_MULT 0x5DEECE66DUL
#define MASK48 0xFFFFFFFFFFFFUL

typedef struct {
    ulong pos_hash;
    ulong condition;
    ulong offset;
    ulong _padding;
} CheckObject;

inline bool check_single(ulong upper_bits, __constant CheckObject* check) {
    ulong result = upper_bits ^ check->pos_hash;
    result = result * JAVA_LCG_MULT;
    result = result + check->offset;
    result = result & MASK48;
    return result < check->condition;
}

__kernel void bedrock_filter(
    ulong start_seed,
    uint batch_size,
    __constant CheckObject* checks,
    uint num_checks,
    __global ulong* candidates,
    __global uint* candidate_count
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    ulong upper_bits = start_seed + ((ulong)gid << 12);

    for (uint i = 0; i < num_checks; i++) {
        if (check_single(upper_bits, &checks[i])) {
            return;  // Сид отклонён
        }
    }

    uint idx = atomic_inc(candidate_count);
    candidates[idx] = upper_bits;
}
```

### Сравнение производительности

| Платформа | Итераций/сек | Время полного поиска |
|-----------|--------------|---------------------|
| CPU (1 поток) | ~50M | ~16 часов |
| CPU (8 потоков) | ~400M | ~2 часа |
| CPU (16 потоков) | ~800M | ~1 час |
| GPU (RTX 3070) | ~10B | ~7 минут |
| GPU (RTX 4080) | ~20B | ~3 минуты |

## Reverse NextLong

Ключевая операция для восстановления сида. Java `nextLong()`:

```java
long nextLong() {
    return ((long)next(32) << 32) + next(32);
}

int next(int bits) {
    seed = (seed * 0x5DEECE66DL + 0xBL) & MASK48;
    return (int)(seed >>> (48 - bits));
}
```

Обратная операция находит возможные 48-битные состояния, которые могли породить данный 64-битный результат. Из-за потери битов при сдвиге, обычно есть 2-4 кандидата.

## Оптимизации

1. **Сортировка по фильтрующей силе** — блоки, отсеивающие больше сидов, проверяются первыми
2. **Батчевая обработка** — проверка множества сидов за один проход
3. **Выбор первичной поверхности** — используется поверхность с лучшей фильтрацией
4. **Ранний выход** — при первом несовпадении сид сразу отклоняется
5. **SIMD/GPU параллелизм** — миллионы проверок одновременно

## Входные данные

Для работы алгоритма нужны координаты блоков с их типом:

```
x: координата X блока
y: координата Y (высота, 0-4 для пола, 123-127 для потолка)
z: координата Z блока
type: BEDROCK или OTHER
```

Рекомендуется указать минимум 20-30 блоков для надёжного поиска.

## Ограничения

1. **Только Nether** — алгоритм работает только для Нижнего мира
2. **Vanilla generation** — не работает с модами, изменяющими генерацию
3. **PaperMC < 1.19.2-213** — требует отдельного режима из-за бага
4. **48-битный сид** — восстанавливается structure seed, для world seed нужна дополнительная работа
