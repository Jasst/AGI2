# merge_files.py — РЕКОМЕНДУЕМЫЙ ВАРИАНТ

def merge_files(input_files, output_file):
    """
    Объединяет несколько файлов в один с обработкой ошибок.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    if not content.endswith('\n'):
                        outfile.write('\n\n')  # Разделитель между файлами
                    # Добавляем комментарий-разделитель для отладки
                    outfile.write(f'\n# >>> END OF {input_file} <<<\n\n')
                print(f"✓ Добавлен: {input_file}")
            except FileNotFoundError:
                print(f"✗ Не найден: {input_file}")
                return False
            except Exception as e:
                print(f"✗ Ошибка {input_file}: {e}")
                return False
    print(f"\n✓ Готово! → {output_file}")
    return True

if __name__ == "__main__":
    files = [
        'emergent_agi_v3.py',
        'emergent_agi_v3_part2.py',
        'emergent_agi_v3_part3.py'
    ]
    merge_files(files, 'emergent_agi_complete.py')