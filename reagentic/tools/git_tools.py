import subprocess
from typing import List, Optional, Union


def call_git_status(short: bool = False) -> str:
    """Возвращает результат `git status`.

    Args:
        short: Если True, использует `-s` для краткого вывода.
    """
    cmd = ['git', 'status', '-s'] if short else ['git', 'status']
    return subprocess.check_output(cmd).decode('utf-8')


def call_git_diff_staged() -> str:
    """Возвращает изменения в staged-файлах (сравнивает с HEAD)."""
    return subprocess.check_output(['git', 'diff', '--cached']).decode('utf-8')


def call_git_diff_unstaged() -> str:
    """Возвращает изменения в unstaged-файлах (сравнивает рабочую директорию и индекс)."""
    return subprocess.check_output(['git', 'diff']).decode('utf-8')


def call_git_diff_all() -> str:
    """Возвращает все изменения (рабочая директория + staged относительно HEAD)."""
    return subprocess.check_output(['git', 'diff', 'HEAD']).decode('utf-8')


def git_add(path: Union[str, List[str]]) -> str:
    """Добавляет файл(ы) в индекс Git.

    Args:
        path: Путь к файлу или список путей. Можно использовать '.' для всех файлов.
    """
    if isinstance(path, str):
        path = [path]
    return subprocess.check_output(['git', 'add'] + path).decode('utf-8')


def git_commit(message: str) -> str:
    """Создает коммит с указанным сообщением."""
    return subprocess.check_output(['git', 'commit', '-m', message]).decode('utf-8')


def git_push(branch: Optional[str] = None, remote: str = 'origin') -> str:
    """Отправляет изменения в удаленный репозиторий.

    Args:
        branch: Ветка для пуша. Если None, используется текущая ветка.
        remote: Имя удаленного репозитория (по умолчанию 'origin').
    """
    cmd = ['git', 'push', remote]
    if branch:
        cmd.append(branch)
    return subprocess.check_output(cmd).decode('utf-8')


def git_pull(remote: str = 'origin', branch: Optional[str] = None) -> str:
    """Забирает изменения из удаленного репозитория.

    Args:
        remote: Имя удаленного репозитория (по умолчанию 'origin').
        branch: Ветка для пулла. Если None, используется текущая ветка.
    """
    cmd = ['git', 'pull', remote]
    if branch:
        cmd.append(branch)
    return subprocess.check_output(cmd).decode('utf-8')


def git_branch_list() -> str:
    """Возвращает список веток."""
    return subprocess.check_output(['git', 'branch']).decode('utf-8')


def git_checkout(branch: str, create: bool = False) -> str:
    """Переключается на указанную ветку.

    Args:
        branch: Имя ветки.
        create: Если True, создает новую ветку.
    """
    cmd = ['git', 'checkout']
    if create:
        cmd.extend(['-b', branch])
    else:
        cmd.append(branch)
    return subprocess.check_output(cmd).decode('utf-8')


def git_log(limit: Optional[int] = None) -> str:
    """Возвращает историю коммитов.

    Args:
        limit: Ограничение количества коммитов.
    """
    cmd = ['git', 'log']
    if limit:
        cmd.extend(['-n', str(limit)])
    return subprocess.check_output(cmd).decode('utf-8')


def git_reset(path: Optional[str] = None, hard: bool = False) -> str:
    """Сбрасывает изменения.

    Args:
        path: Путь к файлу (если None, сбрасывает весь индекс).
        hard: Если True, выполняет жесткий reset (--hard).
    """
    cmd = ['git', 'reset']
    if hard:
        cmd.append('--hard')
    if path:
        cmd.append(path)
    return subprocess.check_output(cmd).decode('utf-8')


def git_remote_list() -> str:
    """Возвращает список удаленных репозиториев."""
    return subprocess.check_output(['git', 'remote', '-v']).decode('utf-8')
