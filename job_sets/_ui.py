from contextlib import contextmanager
from rich.live import Live
from rich.table import Table




class LiveTable:
    def __init__(self, title, columns, create_live_context=True, **table_kwargs):
        """
        Initialize the LiveTable context manager.

        Parameters:
            title (str): Title of the table.
            columns (dict): Dictionary where keys are data keys and values are
                            tuples of (column_name, style), where 'column_name'
                            is the display name of the column and 'style' is the
                            Rich style string for the column.
            **table_kwargs: Additional keyword arguments to pass to the Table constructor.
        """
        self.table = Table(title=title, **table_kwargs)
        self.columns = columns
        for key, (column_name, style) in columns.items():
            self.table.add_column(column_name, style=style, no_wrap=True)
        
        if create_live_context:
            self.live = Live(self.table, refresh_per_second=10)
        else:
            self.live = None

    def __enter__(self):
        if self.live:
            self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.live:
            self.live.__exit__(exc_type, exc_value, traceback)

    def step(self, row_data):
        """
        Add a row to the live table.

        Parameters:
            row_data (dict): Dictionary of data for the new row.
                             Keys should correspond to those specified in columns.
        """
        row = [str(row_data.get(key, "")) for key in self.columns]
        self.table.add_row(*row)
        if self.live:
            self.live.refresh()

def vertical_grid(renderables):
    grid = Table.grid(expand=True)
    for renderable in renderables:
        grid.add_row(renderable)
    return grid

@contextmanager
def chain_live_tables(*tables):
    """
    Chain multiple LiveTable context managers together.

    Parameters:
        *tables (LiveTable): LiveTable instances to chain together.
    """
    with Live(vertical_grid([table.table for table in tables]), refresh_per_second=10) as live:
        yield live
        