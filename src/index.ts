// Мета: автоматично сформувати оптимальний розклад (Resource‑Constrained
// Project Scheduling Problem). Працює з tasks.json та devs.json,
// видає schedule.json і log.json. Запускається через:
//   npm start

import fs from "fs/promises";
import { randomInt } from "crypto";

/* -------------------------------------------------------------------------- */
/*                                Типи даних                                  */
/* -------------------------------------------------------------------------- */

// Опис структури завдання
export interface Task {
    id: string;              // Ідентифікатор (T1, T2 …)
    dur: number;             // Тривалість, год
    deadline: number;        // Дедлайн, Unix‑мс
    skills: string[];        // Необхідні навички
    weight: number;          // Вага у штрафі за просрочку
    deps?: string[];         // ids попередників
}

// Опис структури виконавця
export interface Dev {
    id: string;              // Ім’я/логін
    rate: number;            // $/год
    hoursAvail: number;      // Доступні години (для контролю навантаження)
    skills: string[];        // Компетенції
}

// Один ген — відповідність завдання виконавцю
interface Gene { taskId: string; devId: string; }

// Хромосома — повний розподіл задач
type Chromosome = Gene[];

// Запис у лог історії — значення найкращої функції пристосованості
interface HistoryPoint { generation: number; best: number; }

/* -------------------------- Гіперпараметри GA ----------------------------- */

const POP_SIZE = 50; // Розмір популяції
const GENERATIONS = 50; // Кількість поколінь
const PC = 0.8; // Ймовірність кросоверу
const PM = 0.3; // Ймовірність мутації
const W = { time: 0.5, cost: 0.3, load: 0.2 }; // Ваги критеріїв цільової функції

/* ------------------------- Основна функція алгоритму ---------------------- */

export async function geneticSchedule(tasks: Task[], devs: Dev[]) {
    validate(tasks, devs); // Перевірка вхідних даних

    // Початкова популяція
    let pop: Chromosome[] = Array.from({ length: POP_SIZE }, () => randomChrom(tasks, devs));
    const history: HistoryPoint[] = [];

    for (let g = 0; g < GENERATIONS; g++) {
        pop = nextGeneration(pop, tasks, devs); // Створення нового покоління

        const best = Math.min(...pop.map(ch => fitness(ch, tasks, devs))); // Пошук найкращої хромосоми

        history.push({ generation: g, best }); // Збереження історії
        console.log(`Gen ${g}: best F = ${best.toFixed(2)}`);
    }

    // Вибір чемпіона популяції
    const champion = pop.reduce((a, b) =>
        fitness(a, tasks, devs) < fitness(b, tasks, devs) ? a : b
    );

    return { schedule: decode(champion, tasks, devs), history };
}

/* ------------------------------ CLI запуск -------------------------------- */

if (process.argv.length > 2 && import.meta.url === `file://${process.argv[1]}`) {
    (async () => {
        const [tasksFile, devsFile] = process.argv.slice(2);

        if (!tasksFile || !devsFile) {
            console.error("Usage: ts-node gaScheduler.ts tasks.json devs.json");
            process.exit(1);
        }

        // Читання файлів
        const tasks: Task[] = JSON.parse(await fs.readFile(tasksFile, "utf-8"));
        const devs: Dev[] = JSON.parse(await fs.readFile(devsFile, "utf-8"));
        const { schedule, history } = await geneticSchedule(tasks, devs);

        // Запис результатів у файли
        await fs.writeFile("schedule.json", JSON.stringify(schedule, null, 2));
        await fs.writeFile("log.json", JSON.stringify(history, null, 2));
        console.log("Done. Files saved: schedule.json, log.json");
    })();
}

/* -------------------------- Генерація нового покоління -------------------- */

function nextGeneration(pop: Chromosome[], tasks: Task[], devs: Dev[]): Chromosome[] {
    const offspring: Chromosome[] = [];

    while (offspring.length < POP_SIZE) {
        // Вибір батьків
        const parents: [Chromosome, Chromosome] = [
            tournament(pop, tasks, devs),
            tournament(pop, tasks, devs),
        ];

        // Кросовер або клонування
        let [c1, c2] = Math.random() < PC
            ? crossover(parents[0], parents[1])
            : [parents[0].slice(), parents[1].slice()];

        // Мутації
        if (Math.random() < PM) c1 = mutate(c1, tasks, devs);
        if (Math.random() < PM) c2 = mutate(c2, tasks, devs);

        offspring.push(c1, c2);
    }

    return offspring.slice(0, POP_SIZE); // Обрізання до потрібного розміру
}

// Турнірна селекція — вибір кращого з двох випадкових
function tournament(pop: Chromosome[], tasks: Task[], devs: Dev[]): Chromosome {
    const [a, b] = [pop[randomInt(pop.length)], pop[randomInt(pop.length)]];
    return fitness(a, tasks, devs) < fitness(b, tasks, devs) ? a : b;
}

/* -------------------------- Генерація випадкової хромосоми ---------------- */

function randomChrom(tasks: Task[], devs: Dev[]): Chromosome {
    return tasks.map(t => {
        const pool = devs.filter(d => t.skills.every(s => d.skills.includes(s))); // тільки сумісні
        return { taskId: t.id, devId: pool[randomInt(pool.length)].id };
    });
}

/* ----------------------- Декодування хромосоми у розклад ------------------ */

interface ScheduleItem { label: string; dev: string; start: number; end: number; }
type Schedule = ScheduleItem[];

function decode(ch: Chromosome, tasks: Task[], devs: Dev[]): Schedule {
    const byId = Object.fromEntries(tasks.map(t => [t.id, t]));
    const devFree: Record<string, number> = Object.fromEntries(devs.map(d => [d.id, 0]));
    const done: Record<string, number> = {}; // коли задача завершена
    const sched: Schedule = [];

    // Послідовне розкладання задач
    ch.forEach(g => {
        const t = byId[g.taskId];
        let est = t.deps ? Math.max(...t.deps.map(x => done[x] ?? 0)) : 0; // earliest start time
        est = Math.max(est, devFree[g.devId]); // врахування зайнятості девелопера

        const start = est, end = start + t.dur;

        devFree[g.devId] = end; // оновлення часу зайнятості
        done[t.id] = end;
        sched.push({ label: t.id, dev: g.devId, start, end });
    });

    return sched;
}

/* ---------------------- Функція пристосованості (fitness) ----------------- */

function fitness(ch: Chromosome, tasks: Task[], devs: Dev[]): number {
    const sch = decode(ch, tasks, devs);
    const byLabel = Object.fromEntries(sch.map(s => [s.label, s]));

    const cmax = Math.max(...sch.map(s => s.end)); // makespan
    const cost = sch.reduce((sum, s) => {
        const dev = devs.find(d => d.id === s.dev)!;
        return sum + dev.rate * (s.end - s.start);
    }, 0); // загальні витрати

    // Обчислення завантаженості
    const loadMap: Record<string, number> = {};
    sch.forEach(s => loadMap[s.dev] = (loadMap[s.dev] ?? 0) + (s.end - s.start));

    const loads = Object.values(loadMap);
    const mean = loads.reduce((a, b) => a + b, 0) / loads.length;
    const std  = Math.sqrt(loads.reduce((acc, l) => acc + (l - mean) ** 2, 0) / loads.length); // дисбаланс

    let penalty = 0;

    // Штрафи за невідповідність навичок
    sch.forEach(s => {
        const task = tasks.find(t => t.id === s.label)!;
        const dev  = devs.find(d => d.id === s.dev)!;
        if (!task.skills.every(req => dev.skills.includes(req)))
            penalty += 1_000_000;
    });

    // Штрафи за перевищення доступних годин
    devs.forEach(d => {
        const used = loadMap[d.id] ?? 0;
        if (used > d.hoursAvail)
            penalty += 50_000 * (used - d.hoursAvail);
    });

    // Штрафи за прострочені дедлайни
    tasks.forEach(t => {
        if (byLabel[t.id].end > t.deadline)
            penalty += (byLabel[t.id].end - t.deadline) * t.weight * 10;
    });

    return W.time * cmax + W.cost * cost + W.load * std + penalty;
}

/* ------------------- Кросовер: Order Crossover + Dev shuffle ------------- */

function crossover(p1: Chromosome, p2: Chromosome): [Chromosome, Chromosome] {
    const n = p1.length;
    const [a, b] = [randomInt(n), randomInt(n)].sort((x, y) => x - y); // межі вікна

    const child1: Gene[] = Array(n);
    const child2: Gene[] = Array(n);

    // Копіювання сегмента
    for (let i = a; i < b; i++) {
        child1[i] = { ...p1[i] };
        child2[i] = { ...p2[i] };
    }

    // Доповнення збереженням порядку
    let k1 = b, k2 = b;
    for (let i = 0; i < n; i++) {
        const geneP2 = p2[i];
        if (!child1.some(g => g?.taskId === geneP2.taskId)) {
            child1[k1 % n] = { ...geneP2 }; k1++;
        }

        const geneP1 = p1[i];
        if (!child2.some(g => g?.taskId === geneP1.taskId)) {
            child2[k2 % n] = { ...geneP1 }; k2++;
        }
    }

    // Обмін devId в межах OX-вікна
    for (let i = a; i < b; i++) {
        if (Math.random() < 0.5)
            [child1[i].devId, child2[i].devId] = [child2[i].devId, child1[i].devId];
    }

    return [child1, child2];
}

/* ------------------------------- Мутація ---------------------------------- */

function mutate(ch: Chromosome, tasks: Task[], devs: Dev[]): Chromosome {
    const c = ch.map(g => ({ ...g }));

    if (Math.random() < 0.5) {
        // Обмін порядку завдань
        const [i, j] = [randomInt(c.length), randomInt(c.length)];
        [c[i], c[j]] = [c[j], c[i]];
    } else {
        // Зміна виконавця одного завдання
        const i = randomInt(c.length);
        const task = tasks.find(t => t.id === c[i].taskId)!;
        const pool = devs.filter(d => task.skills.every(s => d.skills.includes(s)));

        c[i].devId = pool[randomInt(pool.length)].id;
    }

    return c;
}

/* ------------------------------ Перевірка даних --------------------------- */

function validate(tasks: Task[], devs: Dev[]) {
    const ids = new Set(tasks.map(t => t.id));

    tasks.forEach(t => {
        t.deps?.forEach(dep => {
            if (!ids.has(dep)) throw new Error(`Unknown dependency ${dep}`);
        });

        if (!devs.some(d => t.skills.every(s => d.skills.includes(s))))
            throw new Error(`No compatible dev for task ${t.id}`);
    });
}
